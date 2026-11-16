from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from database import (  # noqa: E402
    DEFAULT_DATABASE_PATH,
    get_elements,
    get_floor_plan,
    get_uncertainties,
    list_floor_plans,
    open_database,
)
from gemini_assistant import GeminiAssistant, load_api_key, unsupported_reason  # noqa: E402
from query_tools import execute_tool  # noqa: E402


st.set_page_config(
    page_title="Ask Your Floor Plan",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="expanded",
)


def project_path(relative_path: str) -> Path:
    candidate = (PROJECT_ROOT / relative_path).resolve()
    if not candidate.is_relative_to(PROJECT_ROOT):
        raise ValueError("Stored path is outside the project root")
    return candidate


@st.cache_data(show_spinner=False)
def get_plans(database_path: str) -> list[dict[str, Any]]:
    with open_database(database_path, readonly=True) as connection:
        return list_floor_plans(connection)


def get_counts(database_path: str) -> dict[str, int]:
    with open_database(database_path, readonly=True) as connection:
        row = connection.execute(
            """SELECT
                COUNT(DISTINCT fp.image_id) AS floor_plans,
                SUM(CASE WHEN e.category='room' THEN 1 ELSE 0 END) AS rooms,
                SUM(CASE WHEN e.category='wall' THEN 1 ELSE 0 END) AS walls,
                SUM(CASE WHEN e.category='door' THEN 1 ELSE 0 END) AS doors,
                SUM(CASE WHEN e.category='window' THEN 1 ELSE 0 END) AS windows
            FROM floor_plans fp LEFT JOIN elements e ON e.floor_plan_id=fp.image_id"""
        ).fetchone()
    return {key: int(row[key] or 0) for key in ("floor_plans", "rooms", "walls", "doors", "windows")}


def get_plan_elements(database_path: str, image_id: int) -> list[dict[str, Any]]:
    with open_database(database_path, readonly=True) as connection:
        return get_elements(connection, image_id)


def get_plan_record(database_path: str, image_id: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    with open_database(database_path, readonly=True) as connection:
        return get_floor_plan(connection, image_id), get_uncertainties(connection, image_id)


def answer_database_question(
    question: str,
    database_path: str,
    selected_floor_plan_id: int | None = None,
) -> dict[str, Any] | None:
    """Answer recognizable questions directly from approved SQLite query operations."""
    started = time.perf_counter()
    refusal_reason = unsupported_reason(question)
    if refusal_reason:
        try:
            result = execute_tool("get_data_capabilities", {}, database_path)
            return {
                "status": "refused",
                "source": "SQLite capability check",
                "answer": f"I can’t answer that from the stored floor-plan data. {refusal_reason}",
                "tool_calls": [result],
                "supporting_data": result["rows"],
                "floor_plan_id": selected_floor_plan_id,
                "uncertainty": refusal_reason,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        except Exception as error:
            return {
                "status": "error",
                "source": "SQLite capability check",
                "answer": "The database could not be queried, so I cannot answer this unsupported question.",
                "tool_calls": [],
                "supporting_data": [],
                "floor_plan_id": selected_floor_plan_id,
                "uncertainty": f"{type(error).__name__}: {error}",
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }

    lowered = question.lower()
    plan_match = re.search(r"\b(?:floor\s*plan|plan)\s*#?\s*(\d+)\b", lowered)
    image_id = int(plan_match.group(1)) if plan_match else selected_floor_plan_id
    categories = [
        category
        for category, pattern in (
            ("room", r"\brooms?\b"),
            ("wall", r"\bwalls?\b"),
            ("door", r"\bdoors?\b"),
            ("window", r"\bwindows?\b"),
        )
        if re.search(pattern, lowered)
    ]
    count_intent = bool(re.search(r"\b(how many|count|number of|quantity of)\b", lowered))
    operation: str | None = None
    arguments: dict[str, Any] = {}
    render_answer: Any = None

    if count_intent and (categories or re.search(r"\b(candidate )?elements?\b", lowered)):
        if image_id is not None:
            operation = "get_floor_plan_summary"
            arguments = {"floor_plan_id": image_id}

            def render_answer(rows: list[dict[str, Any]]) -> str:
                if not rows:
                    return f"Floor plan {image_id} is not present in the database."
                row = rows[0]
                selected_categories = categories or ["room", "wall", "door", "window"]
                labels = {"room": "rooms", "wall": "walls", "door": "doors", "window": "windows"}
                values = [f"{int(row[f'candidate_{category}s'])} candidate {labels[category]}" for category in selected_categories]
                return f"Plan {image_id}: " + ", ".join(values) + "."
        else:
            operation = "count_elements"
            arguments = {"category": None}

            def render_answer(rows: list[dict[str, Any]]) -> str:
                category_totals = {category: 0 for category in (categories or ["room", "wall", "door", "window"])}
                for row in rows:
                    if row["category"] in category_totals:
                        category_totals[row["category"]] += int(row["candidate_count"])
                labels = {"room": "rooms", "wall": "walls", "door": "doors", "window": "windows"}
                values = [f"{count} candidate {labels[category]}" for category, count in category_totals.items()]
                return "Across all processed plans: " + ", ".join(values) + "."
    elif "largest" in lowered and "room" in lowered:
        operation = "largest_room"
        arguments = {"floor_plan_id": image_id} if image_id is not None else {}

        def render_answer(rows: list[dict[str, Any]]) -> str:
            if not rows:
                return "No candidate room regions were returned for that scope."
            row = rows[0]
            return (
                f"Largest candidate room region: {row['pixel_area']} pixels in plan "
                f"{row['floor_plan_id']} (bounding box x={row['bbox_x']}, y={row['bbox_y']}, "
                f"width={row['bbox_width']}, height={row['bbox_height']})."
            )
    elif "area" in lowered and "room" in lowered:
        operation = "sum_room_pixel_area"
        arguments = {"floor_plan_id": image_id} if image_id is not None else {}

        def render_answer(rows: list[dict[str, Any]]) -> str:
            total = sum(int(row["total_candidate_room_pixels"]) for row in rows)
            return f"Total candidate room area: {total} pixels. No real-world scale is available."
    elif re.search(r"\b(where|location|located|show)\b", lowered) and categories:
        operation = "list_elements"
        arguments = {"floor_plan_id": image_id, "category": categories[0], "limit": 20}

        def render_answer(rows: list[dict[str, Any]]) -> str:
            if not rows:
                return f"No candidate {categories[0]} detections were returned for that scope."
            locations = [
                f"{row['detection_id']}: x={row['bbox_x']}, y={row['bbox_y']}, "
                f"width={row['bbox_width']}, height={row['bbox_height']} px"
                for row in rows
            ]
            return "Candidate locations: " + "; ".join(locations) + "."
    elif "most" in lowered and "room" in lowered:
        operation = "most_rooms"
        arguments = {}

        def render_answer(rows: list[dict[str, Any]]) -> str:
            if not rows:
                return "No candidate room detections are stored."
            leaders = ", ".join(
                f"plan {row['floor_plan_id']} ({row['candidate_room_count']} candidates)"
                for row in rows
            )
            return "Plans with the most candidate rooms: " + leaders + "."
    elif "compare" in lowered and "room" in lowered:
        plan_ids = [int(value) for value in re.findall(r"\b(?:plan|floor\s*plan)\s*#?\s*(\d+)\b", lowered)]
        if len(plan_ids) >= 2:
            operation = "compare_room_counts"
            arguments = {"floor_plan_id_a": plan_ids[0], "floor_plan_id_b": plan_ids[1]}

            def render_answer(rows: list[dict[str, Any]]) -> str:
                return "Candidate room counts: " + ", ".join(
                    f"plan {row['floor_plan_id']}: {row['candidate_room_count']}"
                    for row in rows
                ) + "."

    if operation is None or render_answer is None:
        return None
    try:
        result = execute_tool(operation, arguments, database_path)
        answer = render_answer(result["rows"])
        return {
            "status": "answered" if result["rows"] else "refused",
            "source": "SQLite database-first",
            "answer": answer,
            "tool_calls": [result],
            "supporting_data": result["rows"],
            "floor_plan_id": result.get("floor_plan_id", image_id),
            "uncertainty": result.get("uncertainty") or "All returned detections are candidates with heuristic confidence.",
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }
    except Exception as error:
        return {
            "status": "error",
            "source": "SQLite database-first",
            "answer": "The database query failed; no answer was inferred.",
            "tool_calls": [],
            "supporting_data": [],
            "floor_plan_id": image_id,
            "uncertainty": f"{type(error).__name__}: {error}",
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }


def show_dashboard(plans: list[dict[str, Any]], counts: dict[str, int]) -> None:
    st.title("Ask Your Floor Plan")
    st.caption("Floor-plan exploration using image geometry and unverified mask evidence.")
    columns = st.columns(5)
    labels = (
        ("Processed plans", "floor_plans"),
        ("Candidate rooms", "rooms"),
        ("Candidate walls", "walls"),
        ("Candidate doors", "doors"),
        ("Candidate windows", "windows"),
    )
    for column, (label, key) in zip(columns, labels):
        column.metric(label, f"{counts[key]:,}")

    st.subheader("Available floor plans")
    if plans:
        st.dataframe(
            [
                {
                    "Image ID": plan["image_id"],
                    "Image": plan["image_filename"],
                    "Dimensions": f"{plan['image_width']} × {plan['image_height']} px",
                    "Candidate elements": plan["element_count"],
                    "Status": plan["semantic_status"],
                }
                for plan in plans
            ],
            hide_index=True,
            width="stretch",
        )
    st.info("Every extracted object shown in this prototype is a candidate, not a verified architectural fact.")


def show_explorer(database_path: str, plans: list[dict[str, Any]]) -> None:
    st.title("Floor Plan Explorer")
    if not plans:
        st.warning("No floor plans are in the database. Run the extraction and database import commands first.")
        return
    image_id = st.selectbox("Floor-plan ID", [int(plan["image_id"]) for plan in plans], key="explorer_plan")
    plan, uncertainties = get_plan_record(database_path, image_id)
    if plan is None:
        st.error("The selected plan is not available in SQLite.")
        return

    left, right = st.columns(2)
    original_path = project_path(plan["source_image_path"])
    annotated_path = project_path(plan["annotated_image_path"])
    if original_path.is_file():
        left.image(Image.open(original_path), caption=f"Original · ID {image_id}", width="stretch")
    else:
        left.warning(f"Original image missing: {plan['source_image_path']}")
    if annotated_path.is_file():
        right.image(Image.open(annotated_path), caption="Dashed/color-coded overlays are candidates", width="stretch")
    else:
        right.warning(f"Annotated image missing: {plan['annotated_image_path']}")

    st.caption(f"Dimensions: {plan['image_width']} × {plan['image_height']} pixels. Pixel area is not physical area.")
    elements = get_plan_elements(database_path, image_id)
    for category in ("room", "wall", "door", "window"):
        subset = [element for element in elements if element["category"] == category]
        with st.expander(f"Candidate {category}s · {len(subset)}", expanded=(category == "room")):
            if not subset:
                st.write("No candidate detections for this category in this plan.")
                continue
            st.dataframe(
                [
                    {
                        "ID": element["detection_id"],
                        "Type": element["detection_type"],
                        "Status": element["semantic_status"],
                        "X": element["bbox_x"],
                        "Y": element["bbox_y"],
                        "Width": element["bbox_width"],
                        "Height": element["bbox_height"],
                        "Pixel area": element["pixel_area"],
                        "Confidence (heuristic)": element["confidence"],
                    }
                    for element in subset
                ],
                hide_index=True,
                width="stretch",
            )
            for element in subset[:8]:
                with st.expander(f"Evidence · {element['detection_id']}"):
                    st.write(f"Method: {element['detection_method']}")
                    st.json(json.loads(element["evidence_json"]))

    with st.expander(f"Uncertain or failed detections · {len(uncertainties)}"):
        if uncertainties:
            st.dataframe(uncertainties, hide_index=True, width="stretch")
        else:
            st.write("No specific detector failure was recorded. All semantic detections remain candidates.")


def show_chat(database_path: str, plans: list[dict[str, Any]]) -> None:
    st.title("Ask Your Floor Plan")
    if not plans:
        st.warning("No floor plans are available yet.")
        return
    ids = [int(plan["image_id"]) for plan in plans]
    selected_id = st.selectbox("Selected floor-plan context", [None, *ids], format_func=lambda value: "All plans" if value is None else str(value), key="chat_plan")
    if load_api_key() is None:
        st.info("Gemini is not configured. Add GEMINI_API_KEY to the project .env file to enable live model answers. Database queries and the deterministic evaluation remain available without it.")
    st.caption("Answers must be grounded in stored candidate detections. Pixel measurements are not square metres.")

    messages = st.session_state.setdefault("chat_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("result"):
                result = message["result"]
                if result.get("status") == "error":
                    st.error(result.get("uncertainty") or result["answer"])
                source = result.get("source")
                elapsed = result.get("elapsed_ms")
                meta = []
                if source:
                    meta.append(f"Source: {source}")
                if elapsed is not None:
                    meta.append(f"Response time: {elapsed:.0f} ms")
                if meta:
                    st.caption(" · ".join(meta))
                if result.get("uncertainty"):
                    st.caption(f"Uncertainty: {result['uncertainty']}")
                if result.get("tool_calls"):
                    with st.expander("Database tool call, SQL, and supporting rows"):
                        for tool_call in result["tool_calls"]:
                            st.write(f"Tool: `{tool_call['tool']}`")
                            st.code(tool_call["sql"], language="sql")
                            st.write("Parameters:")
                            st.json(tool_call["parameters"])
                            st.write("Returned rows:")
                            st.json(tool_call["rows"])
                elif result.get("status") == "unconfigured":
                    st.caption("No live Gemini result was generated.")

    question = st.chat_input("Ask about candidate detections, locations, or pixel areas")
    if question:
        messages.append({"role": "user", "content": question})
        with st.spinner("Querying stored candidate detections…"):
            result = answer_database_question(question, database_path, selected_id)
            if result is None:
                result = GeminiAssistant(database_path=database_path).ask(question, selected_id)
        messages.append({"role": "assistant", "content": result["answer"], "result": result})
        st.rerun()


def show_evaluation() -> None:
    st.title("Evaluation")
    result_path = PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_results.json"
    report_path = PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_report.md"
    if not result_path.is_file():
        st.warning("No evaluation run is available. Run `python src/evaluation.py`.")
        return
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    columns = st.columns(5)
    metrics = (
        ("Questions", summary["total_questions"]),
        ("Correct query checks", summary["correct_answers"]),
        ("Incorrect", summary["incorrect_answers"]),
        ("Correct refusals", summary["correct_refusals"]),
        ("Query accuracy", f"{summary['accuracy_on_data_queries']:.1%}"),
    )
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)
    st.caption("This evaluates database operations and deterministic refusals, not live Gemini answer quality.")
    st.dataframe(
        [
            {
                "ID": test["id"],
                "Group": test["group"],
                "Question": test["question"],
                "Operation": test.get("operation", "refusal guard"),
                "Status": test["status"],
            }
            for test in payload["results"]
        ],
        hide_index=True,
        width="stretch",
    )
    if report_path.is_file():
        with st.expander("Evaluation limitations"):
            st.markdown(report_path.read_text(encoding="utf-8").split("## Limitations", 1)[-1])


def main() -> None:
    st.sidebar.title("Ask Your Floor Plan")
    page = st.sidebar.radio("Workspace", ("Dashboard", "Floor Plan Explorer", "Ask Your Floor Plan", "Evaluation"))
    database_path = str(DEFAULT_DATABASE_PATH)
    if not Path(database_path).is_file():
        st.error("SQLite database not found. Run `python src/database.py` from the project root.")
        st.stop()
    plans = get_plans(database_path)
    if page == "Dashboard":
        show_dashboard(plans, get_counts(database_path))
    elif page == "Floor Plan Explorer":
        show_explorer(database_path, plans)
    elif page == "Ask Your Floor Plan":
        show_chat(database_path, plans)
    else:
        show_evaluation()


if __name__ == "__main__":
    main()