from __future__ import annotations

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from database import DEFAULT_DATABASE_PATH, PROJECT_ROOT
from query_tools import TOOL_DECLARATIONS, execute_tool


DEFAULT_MODEL = "gemini-3.8-flash"
MAX_TOOL_ROUNDS = 4
SYSTEM_INSTRUCTION = """You answer only from the provided SQLite tool results. Never use general knowledge
to infer architecture, room names, materials, dimensions, or unobserved objects. Every record
is a preliminary CANDIDATE, with heuristic (not calibrated) confidence. Pixel areas are not
square metres. Bounding-box adjacency is only a geometric estimate. State missing information
and uncertainty explicitly. Do not assign semantics to unassigned mask colors. Do not provide
an answer without calling an approved database tool that supports it; if no tool supports the
question, call get_data_capabilities and decline it. Keep answers concise and cite the floor-plan
ID and detection IDs when present."""


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
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
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

    def ask(self, question: str, selected_floor_plan_id: int | None = None) -> dict[str, Any]:
        if not self.api_key or self.client is None:
            return {
                "status": "unconfigured",
                "answer": "Gemini is not configured. Set GEMINI_API_KEY in the project .env file to enable live question answering.",
                "tool_calls": [],
                "supporting_data": [],
                "floor_plan_id": selected_floor_plan_id,
                "uncertainty": "No live model answer was generated.",
            }

        refused_reason = unsupported_reason(question)
        if refused_reason:
            capability = execute_tool("get_data_capabilities", {}, str(self.database_path))
            return {
                "status": "refused",
                "answer": f"I can’t answer that from the stored floor-plan data. {refused_reason}",
                "tool_calls": [capability],
                "supporting_data": capability["rows"],
                "floor_plan_id": selected_floor_plan_id,
                "uncertainty": refused_reason,
            }

        contents: list[Any] = [
            types.Content(role="user", parts=[types.Part(text=question)])
        ]
        if selected_floor_plan_id is not None:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=f"Selected floor-plan ID: {selected_floor_plan_id}.")],
                )
            )
        tool_results: list[dict[str, Any]] = []
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[self.tool],
            temperature=0.1,
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
                    if name not in {tool["name"] for tool in TOOL_DECLARATIONS}:
                        raise ValueError(f"Gemini requested unapproved function: {name}")
                    result = self._tool_response(name, arguments, selected_floor_plan_id)
                    tool_results.append(result)
                    function_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"result": result},
                        )
                    )
                contents.append(types.Content(role="tool", parts=function_parts))
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
                    answer = "The database returned supporting records, but I could not form a grounded response."
            return {
                "status": status,
                "answer": answer,
                "tool_calls": tool_results,
                "supporting_data": [row for result in tool_results for row in result.get("rows", [])],
                "floor_plan_id": selected_floor_plan_id,
                "uncertainty": uncertainty,
                "model": self.model,
            }
        except Exception as error:
            return {
                "status": "error",
                "answer": "The database-backed Gemini request failed; no unsupported answer was provided.",
                "tool_calls": tool_results,
                "supporting_data": [row for result in tool_results for row in result.get("rows", [])],
                "floor_plan_id": selected_floor_plan_id,
                "uncertainty": sanitize_api_error(error, self.api_key, self.model),
                "model": self.model,
            }


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