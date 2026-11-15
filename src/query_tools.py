from __future__ import annotations

import json
import sqlite3
from typing import Any

from database import open_database


APPROVED_CATEGORIES = {"room", "wall", "door", "window"}
MAX_RESULT_LIMIT = 100

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "list_floor_plans",
        "description": "List processed floor plans and their dimensions/status.",
        "parameters": {"type": "OBJECT", "properties": {}, "required": []},
    },
    {
        "name": "get_floor_plan_summary",
        "description": "Get candidate counts, pixel-area total, and uncertainty count for one floor plan.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"floor_plan_id": {"type": "INTEGER", "description": "Numeric image ID."}},
            "required": ["floor_plan_id"],
        },
    },
    {
        "name": "count_elements",
        "description": "Count stored candidate room, wall, door, or window detections, optionally for one floor plan.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "floor_plan_id": {"type": "INTEGER", "description": "Optional numeric image ID."},
                "category": {"type": "STRING", "enum": ["room", "wall", "door", "window"]},
            },
            "required": [],
        },
    },
    {
        "name": "list_elements",
        "description": "List candidate detections with bounding boxes, pixel areas, confidence and evidence.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "floor_plan_id": {"type": "INTEGER"},
                "category": {"type": "STRING", "enum": ["room", "wall", "door", "window"]},
                "limit": {"type": "INTEGER", "description": "Maximum result count, 1 to 100."},
            },
            "required": [],
        },
    },
    {
        "name": "largest_room",
        "description": "Return the largest candidate room region by pixel area; this is not physical square metres.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"floor_plan_id": {"type": "INTEGER", "description": "Optional numeric image ID."}},
            "required": [],
        },
    },
    {
        "name": "sum_room_pixel_area",
        "description": "Sum candidate room-region pixel areas. No real-world scale is available.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"floor_plan_id": {"type": "INTEGER", "description": "Optional numeric image ID."}},
            "required": [],
        },
    },
    {
        "name": "compare_room_counts",
        "description": "Compare candidate room-region counts between two floor plans.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "floor_plan_id_a": {"type": "INTEGER"},
                "floor_plan_id_b": {"type": "INTEGER"},
            },
            "required": ["floor_plan_id_a", "floor_plan_id_b"],
        },
    },
    {
        "name": "most_rooms",
        "description": "Find the floor plans with the highest number of candidate room regions.",
        "parameters": {"type": "OBJECT", "properties": {}, "required": []},
    },
    {
        "name": "get_element",
        "description": "Retrieve one stored candidate element by its detection ID.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"detection_id": {"type": "STRING"}},
            "required": ["detection_id"],
        },
    },
    {
        "name": "find_adjacent_rooms",
        "description": "Find bounding boxes of candidate room regions that touch or lie within a pixel tolerance; this is a geometric estimate, not verified room adjacency.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "floor_plan_id": {"type": "INTEGER"},
                "tolerance_pixels": {"type": "INTEGER", "description": "0 to 50 pixels."},
            },
            "required": ["floor_plan_id"],
        },
    },
    {
        "name": "get_data_capabilities",
        "description": "Describe what data is and is not stored for these candidate floor-plan detections.",
        "parameters": {"type": "OBJECT", "properties": {}, "required": []},
    },
]


def _rows(connection: sqlite3.Connection, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql, parameters).fetchall()]


def _result(
    operation: str,
    sql: str,
    parameters: tuple[Any, ...],
    rows: list[dict[str, Any]],
    floor_plan_id: int | None = None,
    uncertainty: str | None = None,
) -> dict[str, Any]:
    return {
        "tool": operation,
        "sql": sql,
        "parameters": list(parameters),
        "floor_plan_id": floor_plan_id,
        "rows": rows,
        "uncertainty": uncertainty,
    }


def _require_category(category: Any) -> str:
    if category not in APPROVED_CATEGORIES:
        raise ValueError(f"Unsupported category. Allowed: {sorted(APPROVED_CATEGORIES)}")
    return str(category)


def execute_tool(
    operation: str,
    arguments: dict[str, Any] | None,
    database_path: str,
) -> dict[str, Any]:
    """Execute a registered, parameterized SELECT operation; raw SQL is never accepted."""
    args = arguments or {}
    connection = open_database(database_path, readonly=True)
    try:
        if operation == "list_floor_plans":
            sql = "SELECT image_id, image_filename, image_width, image_height, extraction_status, semantic_status FROM floor_plans ORDER BY image_id"
            return _result(operation, sql, (), _rows(connection, sql))

        if operation == "get_floor_plan_summary":
            image_id = int(args["floor_plan_id"])
            sql = """SELECT fp.image_id, fp.image_width, fp.image_height,
                SUM(CASE WHEN e.category='room' THEN 1 ELSE 0 END) AS candidate_rooms,
                SUM(CASE WHEN e.category='wall' THEN 1 ELSE 0 END) AS candidate_walls,
                SUM(CASE WHEN e.category='door' THEN 1 ELSE 0 END) AS candidate_doors,
                SUM(CASE WHEN e.category='window' THEN 1 ELSE 0 END) AS candidate_windows,
                COALESCE(SUM(CASE WHEN e.category='room' THEN e.pixel_area ELSE 0 END),0) AS room_pixel_area_total,
                (SELECT COUNT(*) FROM uncertainties u WHERE u.floor_plan_id=fp.image_id) AS uncertainty_count
                FROM floor_plans fp LEFT JOIN elements e ON e.floor_plan_id=fp.image_id
                WHERE fp.image_id=? GROUP BY fp.image_id"""
            rows = _rows(connection, sql, (image_id,))
            return _result(operation, sql, (image_id,), rows, image_id, "All element counts are candidate detections.")

        if operation == "count_elements":
            image_id = args.get("floor_plan_id")
            category = args.get("category")
            parameters: list[Any] = []
            predicates: list[str] = []
            if image_id is not None:
                predicates.append("floor_plan_id = ?")
                parameters.append(int(image_id))
            if category is not None:
                predicates.append("category = ?")
                parameters.append(_require_category(category))
            where = " WHERE " + " AND ".join(predicates) if predicates else ""
            sql = "SELECT floor_plan_id, category, COUNT(*) AS candidate_count FROM elements" + where + " GROUP BY floor_plan_id, category ORDER BY floor_plan_id, category"
            rows = _rows(connection, sql, tuple(parameters))
            return _result(operation, sql, tuple(parameters), rows, int(image_id) if image_id is not None else None, "Counts include unverified candidate detections.")

        if operation == "list_elements":
            image_id = args.get("floor_plan_id")
            category = args.get("category")
            limit = max(1, min(int(args.get("limit", 50)), MAX_RESULT_LIMIT))
            predicates: list[str] = []
            parameters: list[Any] = []
            if image_id is not None:
                predicates.append("floor_plan_id = ?")
                parameters.append(int(image_id))
            if category is not None:
                predicates.append("category = ?")
                parameters.append(_require_category(category))
            where = " WHERE " + " AND ".join(predicates) if predicates else ""
            sql = "SELECT detection_id, floor_plan_id, category, detection_type, semantic_status, bbox_x, bbox_y, bbox_width, bbox_height, pixel_area, area_unit, confidence, confidence_kind, detection_method, evidence_json FROM elements" + where + " ORDER BY floor_plan_id, category, bbox_y, bbox_x LIMIT ?"
            parameters.append(limit)
            rows = _rows(connection, sql, tuple(parameters))
            for row in rows:
                row["evidence"] = json.loads(row.pop("evidence_json"))
            return _result(operation, sql, tuple(parameters), rows, int(image_id) if image_id is not None else None, "Bounding boxes and areas are pixel estimates; detections remain candidates.")

        if operation == "largest_room":
            image_id = args.get("floor_plan_id")
            sql = "SELECT detection_id, floor_plan_id, bbox_x, bbox_y, bbox_width, bbox_height, pixel_area, confidence, semantic_status, detection_method, evidence_json FROM elements WHERE category='room' AND (? IS NULL OR floor_plan_id=?) ORDER BY pixel_area DESC LIMIT 1"
            parameters = (int(image_id), int(image_id)) if image_id is not None else (None, None)
            rows = _rows(connection, sql, parameters)
            for row in rows:
                row["evidence"] = json.loads(row.pop("evidence_json"))
            return _result(operation, sql, parameters, rows, int(image_id) if image_id is not None else None, "Largest candidate region by pixel area, not verified room area or square metres.")

        if operation == "sum_room_pixel_area":
            image_id = args.get("floor_plan_id")
            sql = "SELECT floor_plan_id, COUNT(*) AS candidate_room_count, COALESCE(SUM(pixel_area),0) AS total_candidate_room_pixels FROM elements WHERE category='room' AND (? IS NULL OR floor_plan_id=?) GROUP BY floor_plan_id ORDER BY floor_plan_id"
            parameters = (int(image_id), int(image_id)) if image_id is not None else (None, None)
            rows = _rows(connection, sql, parameters)
            return _result(operation, sql, parameters, rows, int(image_id) if image_id is not None else None, "Area is summed in pixels; no physical scale is available and overlapping candidate boxes may double-count geometry.")

        if operation == "compare_room_counts":
            image_a = int(args["floor_plan_id_a"])
            image_b = int(args["floor_plan_id_b"])
            sql = "SELECT floor_plan_id, COUNT(*) AS candidate_room_count FROM elements WHERE category='room' AND floor_plan_id IN (?, ?) GROUP BY floor_plan_id ORDER BY floor_plan_id"
            rows = _rows(connection, sql, (image_a, image_b))
            return _result(operation, sql, (image_a, image_b), rows, uncertainty="Comparison is between candidate-room detections.")

        if operation == "most_rooms":
            sql = "SELECT fp.image_id AS floor_plan_id, COUNT(e.detection_id) AS candidate_room_count FROM floor_plans fp LEFT JOIN elements e ON e.floor_plan_id=fp.image_id AND e.category='room' GROUP BY fp.image_id ORDER BY candidate_room_count DESC, fp.image_id LIMIT 10"
            rows = _rows(connection, sql)
            return _result(operation, sql, (), rows, uncertainty="Ranking uses candidate-room regions, not verified rooms.")

        if operation == "get_element":
            detection_id = str(args["detection_id"])
            sql = "SELECT detection_id, floor_plan_id, category, detection_type, semantic_status, bbox_x, bbox_y, bbox_width, bbox_height, pixel_area, area_unit, confidence, confidence_kind, detection_method, evidence_json, details_json FROM elements WHERE detection_id=?"
            rows = _rows(connection, sql, (detection_id,))
            for row in rows:
                row["evidence"] = json.loads(row.pop("evidence_json"))
                row["details"] = json.loads(row.pop("details_json"))
            plan_id = int(rows[0]["floor_plan_id"]) if rows else None
            return _result(operation, sql, (detection_id,), rows, plan_id, "Returned element is a candidate detection.")

        if operation == "find_adjacent_rooms":
            image_id = int(args["floor_plan_id"])
            tolerance = max(0, min(int(args.get("tolerance_pixels", 5)), 50))
            sql = "SELECT detection_id, floor_plan_id, bbox_x, bbox_y, bbox_width, bbox_height, pixel_area, confidence FROM elements WHERE floor_plan_id=? AND category='room' ORDER BY detection_id"
            parameters = (image_id,)
            rooms = _rows(connection, sql, parameters)
            adjacent = []
            for index, first in enumerate(rooms):
                for second in rooms[index + 1 :]:
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
                    if (gap_x <= tolerance and overlap_y > 0) or (gap_y <= tolerance and overlap_x > 0):
                        adjacent.append(
                            {
                                "element_a": first["detection_id"],
                                "element_b": second["detection_id"],
                                "gap_x_pixels": gap_x,
                                "gap_y_pixels": gap_y,
                                "confidence_a": first["confidence"],
                                "confidence_b": second["confidence"],
                            }
                        )
            return _result(operation, sql, parameters, adjacent, image_id, "Adjacency is estimated from bounding-box proximity; not topological room adjacency.")

        if operation == "get_data_capabilities":
            sql = "SELECT COUNT(*) AS processed_floor_plans FROM floor_plans"
            rows = _rows(connection, sql)
            return _result(
                operation,
                sql,
                (),
                rows,
                uncertainty="Stored areas are pixels only. No room names, physical scale, materials, compass orientation, or authoritative color semantics are stored.",
            )

        raise ValueError(f"Unapproved database operation: {operation}")
    finally:
        connection.close()