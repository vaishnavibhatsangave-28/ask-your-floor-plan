from __future__ import annotations

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from database import DEFAULT_DATABASE_PATH, PROJECT_ROOT
from query_tools import TOOL_DECLARATIONS, execute_tool


DEFAULT_MODEL = "gemini-3.8-flash"
MAX_TOOL_ROUNDS = 3
GEMINI_TIMEOUT_MS = 20_000
SYSTEM_INSTRUCTION = """You answer only from the provided SQLite tool results. Never use general knowledge
to infer architecture, room names, materials, dimensions, or unobserved objects. Every record
is a preliminary CANDIDATE, with heuristic (not calibrated) confidence. Pixel areas are not
square metres. Bounding-box adjacency is only a geometric estimate. State missing information
and uncertainty explicitly. Do not assign semantics to unassigned mask colors. Do not provide
an answer without calling an approved database tool that supports it; if no tool supports the
question, call get_data_capabilities and decline it. Keep answers concise and cite the floor-plan
ID and detection IDs when present."""

_PLAN_IDS = re.compile(
    r"(?:floor[\s-]*plans?|plans?|image(?:\s*id)?)\s+(\d+)"
    r"(?:\s*(?:and|&|versus|vs\.?|/)\s*(?:(?:floor[\s-]*plans?|plans?)\s*)?(\d+))?",
    re.IGNORECASE,
)
_HOW_MANY = re.compile(r"\b(how many|count|number of|detected in)\b", re.IGNORECASE)
_COMPARE = re.compile(
    r"\b(compar(?:e|ison)|which (?:plan|one) has (?:more|fewer)|difference between)\b",
    re.IGNORECASE,
)
_LARGEST = re.compile(r"\blargest\b", re.IGNORECASE)
_PIXEL_AREA = re.compile(r"\b(pixel area|area in pixels|total candidate room area)\b", re.IGNORECASE)
_MOST_ROOMS = re.compile(r"\bmost candidate rooms?\b", re.IGNORECASE)
_ADJACENT = re.compile(r"\badjacent\b", re.IGNORECASE)
_SUMMARY = re.compile(r"\b(summar(?:y|ize)|each category|of each type)\b", re.IGNORECASE)
_LIST_PLANS = re.compile(r"\b(how many floor plans|list (?:the )?floor plans|floor plans (?:are|in) the database)\b", re.IGNORECASE)
_LIST_ELEMENTS = re.compile(r"\b(list|show|bounding box(?:es)?)\b", re.IGNORECASE)
_APPROVED_TOOLS = {tool["name"] for tool in TOOL_DECLARATIONS}


def load_api_key() -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def unsupported_reason(question: str) -> str | None:
    lowered = question.lower()
    checks = (
        (
            r"square\s*(met(er|re)s?|m(?:et(er|re)s?)?)|m²|sqm|real[- ]world (area|size|dimension)|physical (area|dimension)|scale factor",
            "No reliable drawing scale is stored; only pixel areas and pixel bounding boxes are available.",
        ),
        (
            r"room names?|name of (the )?(room|space)|which (bedroom|kitchen|bathroom|living room)|what type of room",
            "Room names and room-type labels are not stored or verified.",
        ),
        (
            r"parking door",
            "Parking doors are not reliably identified in the stored candidate detections.",
        ),
        (
            r"material|wall thickness|north|compass|orientation|construction year|address|owner",
            "That attribute is not present in the stored floor-plan records.",
        ),
        (
            r"what does (the )?(red|214[, ]+70[, ]+70)|meaning of (the )?(red|unknown) color|color.*(mean|represent)|which color.*(wall|door|window|room)",
            "Mask color semantics are unverified; the red color remains unassigned.",
        ),
    )
    for pattern, reason in checks:
        if re.search(pattern, lowered):
            return reason
    return None


def sanitize_api_error(error: Exception, api_key: str | None, model: str) -> str:
    message = str(error)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    lowered = message.lower()
    if "404" in lowered and ("model" in lowered or "not_found" in lowered):
        return f"Gemini model {model} is unavailable to this account or API endpoint."
    if "401" in lowered or "403" in lowered or "api key" in lowered or "unauthenticated" in lowered:
        return "Gemini authentication failed. Verify GEMINI_API_KEY and account access; the key was not displayed."
    return f"Gemini request failed ({type(error).__name__}): {message}"


def extract_plan_ids(question: str) -> list[int]:
    found: list[int] = []
    for match in _PLAN_IDS.finditer(question):
        found.append(int(match.group(1)))
        if match.group(2):
            found.append(int(match.group(2)))
    unique: list[int] = []
    for image_id in found:
        if image_id not in unique:
            unique.append(image_id)
    return unique


def mentioned_categories(question: str) -> list[str]:
    lowered = question.lower()
    found: list[str] = []
    for category, pattern in (
        ("room", r"\brooms?\b"),
        ("wall", r"\bwalls?\b"),
        ("door", r"\bdoors?\b"),
        ("window", r"\bwindows?\b"),
    ):
        if re.search(pattern, lowered):
            found.append(category)
    return found


def resolve_database_intent(
    question: str,
    selected_floor_plan_id: int | None = None,
) -> dict[str, Any] | None:
    """Map a straightforward question onto one registered SQLite tool. Return None if Gemini is needed."""
    plan_ids = extract_plan_ids(question)
    primary_id = plan_ids[0] if plan_ids else selected_floor_plan_id
    categories = mentioned_categories(question)

    if _LIST_PLANS.search(question):
        return {"operation": "list_floor_plans", "arguments": {}}

    if _COMPARE.search(question) and ("room" in categories or "room" in question.lower()):
        if len(plan_ids) >= 2:
            return {
                "operation": "compare_room_counts",
                "arguments": {"floor_plan_id_a": plan_ids[0], "floor_plan_id_b": plan_ids[1]},
            }
        return None

    if _MOST_ROOMS.search(question):
        return {"operation": "most_rooms", "arguments": {}}

    if _ADJACENT.search(question) and primary_id is not None:
        return {
            "operation": "find_adjacent_rooms",
            "arguments": {"floor_plan_id": primary_id, "tolerance_pixels": 5},
        }

    if _LARGEST.search(question) and "room" in categories:
        arguments: dict[str, Any] = {}
        if primary_id is not None:
            arguments["floor_plan_id"] = primary_id
        return {"operation": "largest_room", "arguments": arguments}

    if _PIXEL_AREA.search(question):
        arguments = {}
        if primary_id is not None:
            arguments["floor_plan_id"] = primary_id
        return {"operation": "sum_room_pixel_area", "arguments": arguments}

    if _SUMMARY.search(question) and primary_id is not None:
        return {"operation": "get_floor_plan_summary", "arguments": {"floor_plan_id": primary_id}}

    if _HOW_MANY.search(question) or (len(categories) >= 2 and primary_id is not None):
        if len(categories) >= 2 and primary_id is not None:
            return {"operation": "get_floor_plan_summary", "arguments": {"floor_plan_id": primary_id}}
        if len(categories) == 1:
            arguments = {"category": categories[0]}
            if primary_id is not None:
                arguments["floor_plan_id"] = primary_id
            return {"operation": "count_elements", "arguments": arguments}
        if primary_id is not None and not categories:
            return {"operation": "get_floor_plan_summary", "arguments": {"floor_plan_id": primary_id}}

    if _LIST_ELEMENTS.search(question) and primary_id is not None and categories:
        arguments = {"floor_plan_id": primary_id, "category": categories[0], "limit": 50}
        return {"operation": "list_elements", "arguments": arguments}

    return None


def format_database_answer(result: dict[str, Any]) -> str:
    """Build a concise answer from actual SQLite rows. Values are never hardcoded."""
    rows = result.get("rows") or []
    tool = result.get("tool")
    uncertainty = result.get("uncertainty") or ""

    def with_uncertainty(text: str) -> str:
        return f"{text} {uncertainty}".strip()

    if not rows:
        return with_uncertainty("No matching candidate records were returned by the database query.")

    if tool == "get_floor_plan_summary":
        row = rows[0]
        return with_uncertainty(
            f"Floor plan {int(row['image_id'])} has {int(row['candidate_rooms'] or 0)} candidate rooms, "
            f"{int(row['candidate_walls'] or 0)} candidate walls, {int(row['candidate_doors'] or 0)} candidate doors, "
            f"and {int(row['candidate_windows'] or 0)} candidate windows."
        )

    if tool == "count_elements":
        parts = [
            f"floor plan {int(row['floor_plan_id'])} has {int(row['candidate_count'])} candidate {row['category']} detections"
            for row in rows
        ]
        return with_uncertainty("Returned counts: " + "; ".join(parts) + ".")

    if tool == "compare_room_counts":
        parts = [
            f"floor plan {int(row['floor_plan_id'])} has {int(row['candidate_room_count'])} candidate rooms"
            for row in rows
        ]
        return with_uncertainty("Candidate room comparison: " + "; ".join(parts) + ".")

    if tool == "list_floor_plans":
        return with_uncertainty(f"The database contains {len(rows)} processed floor plans.")

    if tool == "largest_room":
        row = rows[0]
        return with_uncertainty(
            f"The largest candidate room region in floor plan {int(row['floor_plan_id'])} is "
            f"{row['detection_id']} with {int(row['pixel_area'])} pixels."
        )

    if tool == "sum_room_pixel_area":
        if len(rows) == 1:
            row = rows[0]
            return with_uncertainty(
                f"Floor plan {int(row['floor_plan_id'])} has {int(row['candidate_room_count'])} candidate rooms "
                f"totalling {int(row['total_candidate_room_pixels'])} pixels."
            )
        total_pixels = sum(int(row["total_candidate_room_pixels"]) for row in rows)
        total_rooms = sum(int(row["candidate_room_count"]) for row in rows)
        return with_uncertainty(
            f"Across {len(rows)} floor plans there are {total_rooms} candidate rooms totalling {total_pixels} pixels."
        )

    if tool == "most_rooms":
        top = rows[0]
        return with_uncertainty(
            f"Floor plan {int(top['floor_plan_id'])} has the most candidate rooms "
            f"({int(top['candidate_room_count'])})."
        )

    if tool == "get_element":
        row = rows[0]
        return with_uncertainty(
            f"Candidate {row['detection_id']} is a {row['category']} on floor plan {int(row['floor_plan_id'])} "
            f"at bbox ({row['bbox_x']}, {row['bbox_y']}, {row['bbox_width']}, {row['bbox_height']}) "
            f"with pixel area {int(row['pixel_area'])}."
        )

    if tool == "list_elements":
        return with_uncertainty(f"The query returned {len(rows)} candidate detections.")

    if tool == "find_adjacent_rooms":
        plan = result.get("floor_plan_id")
        scope = f" in floor plan {plan}" if plan is not None else ""
        return with_uncertainty(
            f"Bounding-box adjacency found {len(rows)} candidate-room pairs{scope}."
        )

    if tool == "get_data_capabilities":
        return with_uncertainty("These records are candidate detections only.")

    return with_uncertainty("The database returned supporting rows for this question.")


class GeminiAssistant:
    def __init__(
        self,
        database_path: Path | str = DEFAULT_DATABASE_PATH,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.database_path = Path(database_path)
        self.api_key = api_key if api_key is not None else load_api_key()
        self.model = os.getenv("GEMINI_MODEL", model) or DEFAULT_MODEL
        self.client = (
            genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=GEMINI_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            if self.api_key
            else None
        )
        self.tool = types.Tool(
            function_declarations=[types.FunctionDeclaration(**declaration) for declaration in TOOL_DECLARATIONS]
        )

    def test_connection(self) -> dict[str, Any]:
        if not self.api_key or self.client is None:
            return {
                "status": "unconfigured",
                "model": self.model,
                "message": "GEMINI_API_KEY is not configured.",
            }
        try:
            model_info = self.client.models.get(model=self.model)
        except Exception as error:
            return {
                "status": "error",
                "model": self.model,
                "model_accessible": False,
                "message": sanitize_api_error(error, self.api_key, self.model),
            }
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents="Reply with the single word OK.",
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=64),
            )
            return {
                "status": "ok" if response.text else "empty_response",
                "model": self.model,
                "model_accessible": True,
                "supported_actions": list(getattr(model_info, "supported_actions", []) or []),
                "response_received": bool(response.text),
                "finish_reason": (
                    str(response.candidates[0].finish_reason)
                    if response.candidates
                    else None
                ),
            }
        except Exception as error:
            return {
                "status": "generation_error",
                "model": self.model,
                "model_accessible": True,
                "message": sanitize_api_error(error, self.api_key, self.model),
            }

    def _tool_response(
        self,
        operation: str,
        arguments: dict[str, Any],
        selected_floor_plan_id: int | None,
    ) -> dict[str, Any]:
        call_arguments = dict(arguments)
        if selected_floor_plan_id is not None and "floor_plan_id" not in call_arguments:
            if operation in {
                "get_floor_plan_summary",
                "find_adjacent_rooms",
            }:
                call_arguments["floor_plan_id"] = selected_floor_plan_id
            elif operation in {"count_elements", "list_elements", "largest_room", "sum_room_pixel_area"}:
                call_arguments["floor_plan_id"] = selected_floor_plan_id
        return execute_tool(operation, call_arguments, str(self.database_path))

    def _payload(
        self,
        *,
        status: str,
        answer: str,
        tool_results: list[dict[str, Any]],
        selected_floor_plan_id: int | None,
        uncertainty: str,
        source: str,
        started: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "answer": answer,
            "tool_calls": tool_results,
            "supporting_data": [row for result in tool_results for row in result.get("rows", [])],
            "floor_plan_id": selected_floor_plan_id,
            "uncertainty": uncertainty,
            "model": self.model,
            "source": source,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if extra:
            payload.update(extra)
        return payload

    def ask(self, question: str, selected_floor_plan_id: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        refused_reason = unsupported_reason(question)
        if refused_reason:
            capability = execute_tool("get_data_capabilities", {}, str(self.database_path))
            return self._payload(
                status="refused",
                answer=f"I can’t answer that from the stored floor-plan data. {refused_reason}",
                tool_results=[capability],
                selected_floor_plan_id=selected_floor_plan_id,
                uncertainty=refused_reason,
                source="database",
                started=started,
            )

        intent = resolve_database_intent(question, selected_floor_plan_id)
        if intent is not None:
            result = self._tool_response(intent["operation"], intent["arguments"], selected_floor_plan_id)
            has_rows = bool(result.get("rows"))
            return self._payload(
                status="answered" if has_rows else "refused",
                answer=format_database_answer(result),
                tool_results=[result],
                selected_floor_plan_id=selected_floor_plan_id,
                uncertainty=str(result.get("uncertainty") or "All retrieved detections remain candidates with heuristic confidence."),
                source="database",
                started=started,
            )

        if not self.api_key or self.client is None:
            return self._payload(
                status="unconfigured",
                answer="Gemini is not configured. Set GEMINI_API_KEY in the project .env file to enable live question answering.",
                tool_results=[],
                selected_floor_plan_id=selected_floor_plan_id,
                uncertainty="No live model answer was generated.",
                source="gemini",
                started=started,
            )

        user_text = question.strip()
        if selected_floor_plan_id is not None:
            user_text = f"{user_text}\nSelected floor-plan ID: {selected_floor_plan_id}."
        contents: list[Any] = [types.Content(role="user", parts=[types.Part(text=user_text)])]
        tool_results: list[dict[str, Any]] = []
        seen_calls: set[str] = set()
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[self.tool],
            temperature=0.1,
            max_output_tokens=512,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
                maximum_remote_calls=MAX_TOOL_ROUNDS,
            ),
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            for _ in range(MAX_TOOL_ROUNDS):
                calls = response.function_calls or []
                if not calls:
                    break
                if response.candidates:
                    contents.append(response.candidates[0].content)
                function_parts = []
                for function_call in calls:
                    name = str(function_call.name or "")
                    arguments = dict(function_call.args or {})
                    if name not in _APPROVED_TOOLS:
                        raise ValueError(f"Gemini requested unapproved function: {name}")
                    call_key = json.dumps({"name": name, "arguments": arguments}, sort_keys=True)
                    if call_key in seen_calls:
                        continue
                    seen_calls.add(call_key)
                    result = self._tool_response(name, arguments, selected_floor_plan_id)
                    tool_results.append(result)
                    function_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"result": result},
                        )
                    )
                if not function_parts:
                    break
                # Gemini only accepts roles "user" and "model". Function results must be user parts.
                contents.append(types.Content(role="user", parts=function_parts))
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

            answer = (response.text or "").strip()
            has_supporting_rows = any(
                result.get("tool") != "get_data_capabilities" and result.get("rows")
                for result in tool_results
            )
            if not tool_results or not has_supporting_rows:
                capability = execute_tool("get_data_capabilities", {}, str(self.database_path))
                tool_results.append(capability)
                answer = (
                    "I can’t find supporting records for that question in the selected database scope, "
                    "so I’m declining rather than inferring architectural information."
                )
                status = "refused"
                uncertainty = "No matching result rows were returned by an approved data-backed tool."
            else:
                status = "answered" if answer else "refused"
                uncertainty = "; ".join(
                    str(result["uncertainty"])
                    for result in tool_results
                    if result.get("uncertainty")
                ) or "All retrieved detections remain candidates with heuristic confidence."
                if not answer:
                    answer = format_database_answer(tool_results[0])
                    status = "answered"
            return self._payload(
                status=status,
                answer=answer,
                tool_results=tool_results,
                selected_floor_plan_id=selected_floor_plan_id,
                uncertainty=uncertainty,
                source="gemini",
                started=started,
            )
        except Exception as error:
            gemini_error = sanitize_api_error(error, self.api_key, self.model)
            supporting = [result for result in tool_results if result.get("rows") and result.get("tool") != "get_data_capabilities"]
            if supporting:
                return self._payload(
                    status="answered",
                    answer=format_database_answer(supporting[0]),
                    tool_results=tool_results,
                    selected_floor_plan_id=selected_floor_plan_id,
                    uncertainty=f"Gemini failed after the database query succeeded. {gemini_error}",
                    source="database",
                    started=started,
                    extra={"gemini_error": gemini_error},
                )
            return self._payload(
                status="error",
                answer="The database-backed Gemini request failed; no unsupported answer was provided.",
                tool_results=tool_results,
                selected_floor_plan_id=selected_floor_plan_id,
                uncertainty=gemini_error,
                source="gemini",
                started=started,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test connectivity to the configured Gemini model.")
    parser.add_argument("--model", default=None, help="Optional model override for this test only.")
    args = parser.parse_args()
    assistant = GeminiAssistant(model=args.model or DEFAULT_MODEL)
    result = assistant.test_connection()
    print(
        f"status={result['status']} model={result['model']} "
        f"model_accessible={result.get('model_accessible', False)}"
    )
    if result["status"] == "ok":
        print("Gemini returned a response; no credential data was printed.")
    elif result.get("message"):
        print(result["message"])
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
