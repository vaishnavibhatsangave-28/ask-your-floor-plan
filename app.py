from __future__ import annotations

import json
import sys
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
from gemini_assistant import DEFAULT_MODEL, GeminiAssistant  # noqa: E402


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
    assistant = GeminiAssistant(database_path=database_path)
    if assistant.api_key is None:
        st.info("Gemini is not configured. Add GEMINI_API_KEY to the project .env file to enable live model answers. Database queries and the deterministic evaluation remain available without it.")
    st.caption("Answers must be grounded in stored candidate detections. Pixel measurements are not square metres.")

    messages = st.session_state.setdefault("chat_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("result"):
                result = message["result"]
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
        result = assistant.ask(question, selected_id)
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