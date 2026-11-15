from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from database import DEFAULT_DATABASE_PATH, open_database
from gemini_assistant import unsupported_reason
from query_tools import execute_tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def _bbox_distance(first: dict[str, Any], second: dict[str, Any], tolerance: int) -> bool:
    gap_x = max(
        0,
        second["bbox_x"] - (first["bbox_x"] + first["bbox_width"]),
        first["bbox_x"] - (second["bbox_x"] + second["bbox_width"]),
    )
    gap_y = max(
        0,
        second["bbox_y"] - (first["bbox_y"] + first["bbox_height"]),
        first["bbox_y"] - (second["bbox_y"] + second["bbox_height"]),
    )
    overlap_x = min(first["bbox_x"] + first["bbox_width"], second["bbox_x"] + second["bbox_width"]) - max(first["bbox_x"], second["bbox_x"])
    overlap_y = min(first["bbox_y"] + first["bbox_height"], second["bbox_y"] + second["bbox_height"]) - max(first["bbox_y"], second["bbox_y"])
    return (gap_x <= tolerance and overlap_y > 0) or (gap_y <= tolerance and overlap_x > 0)


def build_evaluation_cases(database_path: Path) -> list[dict[str, Any]]:
    with open_database(database_path, readonly=True) as connection:
        plans = [dict(row) for row in connection.execute("SELECT * FROM floor_plans ORDER BY image_id")]
        elements = [dict(row) for row in connection.execute("SELECT * FROM elements ORDER BY floor_plan_id, category, bbox_y, bbox_x")]
        uncertainty_rows = [dict(row) for row in connection.execute("SELECT floor_plan_id, item, reason FROM uncertainties")]

    plan_ids = {int(plan["image_id"]) for plan in plans}
    required_ids = {8, 31, 61}
    if not required_ids.issubset(plan_ids):
        raise ValueError(f"Evaluation requires imported pilot plans {sorted(required_ids)}")
    by_plan: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in plan_ids}
    for element in elements:
        by_plan[int(element["floor_plan_id"])].append(element)
    room_counts = {
        image_id: sum(element["category"] == "room" for element in plan_elements)
        for image_id, plan_elements in by_plan.items()
    }
    rooms_8 = [element for element in by_plan[8] if element["category"] == "room"]
    rooms_31 = [element for element in by_plan[31] if element["category"] == "room"]
    largest_8 = max(rooms_8, key=lambda element: element["pixel_area"])
    largest_31 = max(rooms_31, key=lambda element: element["pixel_area"])
    compare_expected = [
        {"floor_plan_id": image_id, "candidate_room_count": room_counts[image_id]}
        for image_id in (8, 31)
        if room_counts[image_id] > 0
    ]
    top_rooms = [
        {"floor_plan_id": image_id, "candidate_room_count": count}
        for image_id, count in sorted(room_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    door_ids_61 = sorted(
        element["detection_id"]
        for element in by_plan[61]
        if element["category"] == "door"
    )
    largest_room_all = max(
        (element for element in elements if element["category"] == "room"),
        key=lambda element: element["pixel_area"],
    )
    adjacency_rooms = [element for element in rooms_31]
    expected_adjacency = [
        {"element_a": first["detection_id"], "element_b": second["detection_id"]}
        for index, first in enumerate(adjacency_rooms)
        for second in adjacency_rooms[index + 1 :]
        if _bbox_distance(first, second, 5)
    ]
    candidate_counts_61 = {
        category: sum(element["category"] == category for element in by_plan[61])
        for category in ("room", "wall", "door", "window")
    }
    room8_first = sorted(rooms_8, key=lambda element: (element["bbox_y"], element["bbox_x"]))[0]
    room_area_61 = sum(
        int(element["pixel_area"])
        for element in by_plan[61]
        if element["category"] == "room"
    )
    total_room_area = sum(int(element["pixel_area"]) for element in elements if element["category"] == "room")
    uncertainty_counts = {image_id: 0 for image_id in plan_ids}
    for row in uncertainty_rows:
        uncertainty_counts[int(row["floor_plan_id"])] += 1
    summary_61 = {
        "image_id": 61,
        "candidate_rooms": candidate_counts_61["room"],
        "candidate_walls": candidate_counts_61["wall"],
        "candidate_doors": candidate_counts_61["door"],
        "candidate_windows": candidate_counts_61["window"],
        "room_pixel_area_total": room_area_61,
        "uncertainty_count": uncertainty_counts[61],
    }

    return [
        {"id": "E01", "group": "cross-plan count", "question": "How many floor plans are in the database?", "operation": "list_floor_plans", "arguments": {}, "expected": {"floor_plan_ids": sorted(plan_ids)}, "comparison": "floor_plan_ids"},
        {"id": "E02", "group": "count", "question": "How many candidate rooms are detected in plan 8?", "operation": "count_elements", "arguments": {"floor_plan_id": 8, "category": "room"}, "expected": {"candidate_count": room_counts[8]}, "comparison": "single_count"},
        {"id": "E03", "group": "count", "question": "How many candidate doors are in plan 31?", "operation": "count_elements", "arguments": {"floor_plan_id": 31, "category": "door"}, "expected": {"candidate_count": sum(element["category"] == "door" for element in by_plan[31])}, "comparison": "single_count"},
        {"id": "E04", "group": "count", "question": "How many candidate windows are in plan 61?", "operation": "count_elements", "arguments": {"floor_plan_id": 61, "category": "window"}, "expected": {"candidate_count": candidate_counts_61["window"]}, "comparison": "single_count"},
        {"id": "E05", "group": "count", "question": "How many candidate walls are in plan 8?", "operation": "count_elements", "arguments": {"floor_plan_id": 8, "category": "wall"}, "expected": {"candidate_count": sum(element["category"] == "wall" for element in by_plan[8])}, "comparison": "single_count"},
        {"id": "E06", "group": "area", "question": "What is the largest candidate room region in plan 8?", "operation": "largest_room", "arguments": {"floor_plan_id": 8}, "expected": {"detection_id": largest_8["detection_id"], "pixel_area": int(largest_8["pixel_area"])}, "comparison": "largest"},
        {"id": "E07", "group": "area", "question": "What is the largest candidate room region in plan 31?", "operation": "largest_room", "arguments": {"floor_plan_id": 31}, "expected": {"detection_id": largest_31["detection_id"], "pixel_area": int(largest_31["pixel_area"])}, "comparison": "largest"},
        {"id": "E08", "group": "area", "question": "What is the total candidate room area in pixels for plan 61?", "operation": "sum_room_pixel_area", "arguments": {"floor_plan_id": 61}, "expected": {"floor_plan_id": 61, "candidate_room_count": candidate_counts_61["room"], "total_candidate_room_pixels": room_area_61}, "comparison": "sum"},
        {"id": "E09", "group": "area", "question": "What is the total candidate room area in pixels across all plans?", "operation": "sum_room_pixel_area", "arguments": {}, "expected": {"total_candidate_room_pixels": total_room_area, "candidate_room_count": sum(room_counts.values())}, "comparison": "sum_all"},
        {"id": "E10", "group": "comparison", "question": "Compare candidate room counts in plans 8 and 31.", "operation": "compare_room_counts", "arguments": {"floor_plan_id_a": 8, "floor_plan_id_b": 31}, "expected": {"counts": compare_expected}, "comparison": "compare"},
        {"id": "E11", "group": "cross-plan comparison", "question": "Which plans have the most candidate room regions?", "operation": "most_rooms", "arguments": {}, "expected": {"top_plans": top_rooms}, "comparison": "top"},
        {"id": "E12", "group": "location", "question": "Show the bounding box of a candidate room in plan 8.", "operation": "get_element", "arguments": {"detection_id": room8_first["detection_id"]}, "expected": {"detection_id": room8_first["detection_id"], "floor_plan_id": 8, "bbox_x": room8_first["bbox_x"], "bbox_y": room8_first["bbox_y"], "bbox_width": room8_first["bbox_width"], "bbox_height": room8_first["bbox_height"]}, "comparison": "element"},
        {"id": "E13", "group": "summary", "question": "Summarize candidate detections in plan 61.", "operation": "get_floor_plan_summary", "arguments": {"floor_plan_id": 61}, "expected": summary_61, "comparison": "summary"},
        {"id": "E14", "group": "geometry", "question": "Which candidate room boxes are adjacent in plan 31?", "operation": "find_adjacent_rooms", "arguments": {"floor_plan_id": 31, "tolerance_pixels": 5}, "expected": {"pairs": expected_adjacency}, "comparison": "adjacency"},
        {"id": "E15", "group": "count", "question": "How many candidate elements of each category are in plan 61?", "operation": "get_floor_plan_summary", "arguments": {"floor_plan_id": 61}, "expected": summary_61, "comparison": "categories"},
        {"id": "E16", "group": "unsupported", "question": "What is the total room area in square metres for plan 8?", "refusal": True},
        {"id": "E17", "group": "unsupported", "question": "What are the room names and which one is the kitchen?", "refusal": True},
        {"id": "E18", "group": "unsupported", "question": "How many parking doors are in plan 31?", "refusal": True},
        {"id": "E19", "group": "unsupported", "question": "What material are the walls made from?", "refusal": True},
        {"id": "E20", "group": "unsupported", "question": "What architectural class does the red mask color represent?", "refusal": True},
    ]


def _actual_projection(test: dict[str, Any], result: dict[str, Any]) -> Any:
    rows = result["rows"]
    comparison = test["comparison"]
    if comparison == "floor_plan_ids":
        return {"floor_plan_ids": [int(row["image_id"]) for row in rows]}
    if comparison == "single_count":
        return {"candidate_count": int(rows[0]["candidate_count"]) if rows else 0}
    if comparison == "largest":
        return {"detection_id": rows[0]["detection_id"], "pixel_area": int(rows[0]["pixel_area"])} if rows else None
    if comparison == "sum":
        return {
            "floor_plan_id": int(rows[0]["floor_plan_id"]) if rows else test["arguments"]["floor_plan_id"],
            "candidate_room_count": int(rows[0]["candidate_room_count"]) if rows else 0,
            "total_candidate_room_pixels": int(rows[0]["total_candidate_room_pixels"]) if rows else 0,
        }
    if comparison == "sum_all":
        return {
            "candidate_room_count": sum(int(row["candidate_room_count"]) for row in rows),
            "total_candidate_room_pixels": sum(int(row["total_candidate_room_pixels"]) for row in rows),
        }
    if comparison == "compare":
        return {"counts": [{"floor_plan_id": int(row["floor_plan_id"]), "candidate_room_count": int(row["candidate_room_count"])} for row in rows]}
    if comparison == "top":
        return {"top_plans": [{"floor_plan_id": int(row["floor_plan_id"]), "candidate_room_count": int(row["candidate_room_count"])} for row in rows]}
    if comparison == "element":
        if not rows:
            return None
        row = rows[0]
        return {key: int(row[key]) if key != "detection_id" else row[key] for key in ("detection_id", "floor_plan_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height")}
    if comparison in {"summary", "categories"}:
        if not rows:
            return None
        row = rows[0]
        return {
            "image_id": int(row["image_id"]),
            "candidate_rooms": int(row["candidate_rooms"]),
            "candidate_walls": int(row["candidate_walls"]),
            "candidate_doors": int(row["candidate_doors"]),
            "candidate_windows": int(row["candidate_windows"]),
            "room_pixel_area_total": int(row["room_pixel_area_total"]),
            "uncertainty_count": int(row["uncertainty_count"]),
        }
    if comparison == "adjacency":
        return {"pairs": [{"element_a": row["element_a"], "element_b": row["element_b"]} for row in rows]}
    raise ValueError(f"Unknown evaluation comparison: {comparison}")


def run_evaluation(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_evaluation_cases(Path(database_path))
    results: list[dict[str, Any]] = []
    for case in cases:
        if case.get("refusal"):
            reason = unsupported_reason(case["question"])
            capability = execute_tool("get_data_capabilities", {}, str(database_path))
            correct = reason is not None and bool(capability["rows"])
            results.append(
                {
                    **case,
                    "expected_action": "refuse",
                    "actual_action": "refuse" if correct else "unsupported_guard_missing",
                    "status": "correct_refusal" if correct else "failed",
                    "actual": {"reason": reason, "database_tool": capability},
                    "live_gemini_tested": False,
                }
            )
            continue

        result = execute_tool(case["operation"], case["arguments"], str(database_path))
        actual = _actual_projection(case, result)
        expected = case["expected"]
        passed = actual == expected
        results.append(
            {
                **case,
                "status": "passed" if passed else "failed",
                "actual": actual,
                "expected": expected,
                "sql": result["sql"],
                "parameters": result["parameters"],
                "returned_data": result["rows"],
                "uncertainty": result["uncertainty"],
                "live_gemini_tested": False,
            }
        )

    evaluable = [result for result in results if not result.get("refusal")]
    correct = sum(result["status"] == "passed" for result in evaluable)
    incorrect = sum(result["status"] == "failed" for result in evaluable)
    partial = sum(result["status"] == "partially_correct" for result in evaluable)
    correct_refusals = sum(result["status"] == "correct_refusal" for result in results)
    summary = {
        "total_questions": len(results),
        "correct_answers": correct,
        "incorrect_answers": incorrect,
        "partially_correct_answers": partial,
        "correct_refusals": correct_refusals,
        "not_evaluable": sum(result["status"] == "not_evaluable" for result in results),
        "accuracy_on_data_queries": correct / len(evaluable) if evaluable else None,
        "live_gemini_tests": 0,
        "evaluation_scope": "SQLite query operations and deterministic unsupported-question guard; not live Gemini answer quality.",
    }
    payload = {"summary": summary, "results": results}
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "evaluation_report.md", summary, results)
    return payload


def _write_report(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    accuracy = summary["accuracy_on_data_queries"]
    with path.open("w", encoding="utf-8") as report:
        report.write("# 20-Question Evaluation\n\n")
        report.write("## Summary\n\n")
        report.write(f"- Total questions: {summary['total_questions']}\n")
        report.write(f"- Correct database-query answers: {summary['correct_answers']}\n")
        report.write(f"- Incorrect database-query answers: {summary['incorrect_answers']}\n")
        report.write(f"- Partially correct answers: {summary['partially_correct_answers']}\n")
        report.write(f"- Correct unsupported-question refusals: {summary['correct_refusals']}\n")
        report.write(f"- Accuracy on evaluable database queries: {accuracy:.1%}\n" if accuracy is not None else "- Accuracy: not available\n")
        report.write("- Live Gemini answers evaluated: 0 (no API key configured).\n\n")
        report.write(
            "Expected values are computed from the actual imported SQLite records. The five refusal tests exercise the deterministic data-capability guard, not a live Gemini response. Candidate counts and areas are not ground-truth accuracy measurements.\n\n"
        )
        report.write("## Per-question results\n\n")
        report.write("| ID | Group | Question | Operation | Status |\n|---|---|---|---|---|\n")
        for result in results:
            report.write(
                f"| {result['id']} | {result['group']} | {result['question']} | "
                f"{result.get('operation', 'refusal guard')} | {result['status']} |\n"
            )
        report.write("\n## Limitations\n\n")
        report.write(
            "The evaluation verifies database query correctness and deterministic refusals only. It does not evaluate natural-language accuracy of Gemini because GEMINI_API_KEY is not configured. No expected semantic labels or physical areas were invented; the evaluated data are candidate detections and pixel measurements.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible 20-question DB evaluation.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = run_evaluation(args.database, args.output_dir)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()