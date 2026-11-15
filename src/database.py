from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "database" / "ask_your_floor_plan.sqlite3"
DEFAULT_JSON_DIR = PROJECT_ROOT / "outputs" / "extraction" / "json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw" / "archive"
DEFAULT_ANNOTATED_DIR = PROJECT_ROOT / "outputs" / "extraction" / "annotated"


class ClosingConnection(sqlite3.Connection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._context_depth = 0

    def __enter__(self) -> ClosingConnection:
        self._context_depth += 1
        return super().__enter__()

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> Any:
        try:
            return super().__exit__(exception_type, exception, traceback)
        finally:
            self._context_depth -= 1
            if self._context_depth == 0:
                self.close()


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS floor_plans (
    image_id INTEGER PRIMARY KEY,
    image_filename TEXT NOT NULL,
    mask_filename TEXT NOT NULL,
    source_image_path TEXT NOT NULL,
    source_mask_path TEXT NOT NULL,
    annotated_image_path TEXT NOT NULL,
    extraction_json_path TEXT NOT NULL,
    image_width INTEGER NOT NULL CHECK (image_width > 0),
    image_height INTEGER NOT NULL CHECK (image_height > 0),
    extraction_status TEXT NOT NULL DEFAULT 'candidate_extraction',
    semantic_status TEXT NOT NULL DEFAULT 'CANDIDATE',
    confidence_interpretation TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS elements (
    detection_id TEXT PRIMARY KEY,
    floor_plan_id INTEGER NOT NULL REFERENCES floor_plans(image_id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (category IN ('room', 'wall', 'door', 'window')),
    detection_type TEXT NOT NULL,
    semantic_status TEXT NOT NULL CHECK (semantic_status IN ('CANDIDATE', 'CONFIRMED', 'UNKNOWN')),
    bbox_x INTEGER NOT NULL CHECK (bbox_x >= 0),
    bbox_y INTEGER NOT NULL CHECK (bbox_y >= 0),
    bbox_width INTEGER NOT NULL CHECK (bbox_width >= 0),
    bbox_height INTEGER NOT NULL CHECK (bbox_height >= 0),
    pixel_area INTEGER NOT NULL CHECK (pixel_area >= 0),
    area_unit TEXT NOT NULL DEFAULT 'pixels',
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    confidence_kind TEXT NOT NULL DEFAULT 'heuristic_not_calibrated',
    detection_method TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    details_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uncertainties (
    uncertainty_id INTEGER PRIMARY KEY AUTOINCREMENT,
    floor_plan_id INTEGER NOT NULL REFERENCES floor_plans(image_id) ON DELETE CASCADE,
    item TEXT NOT NULL,
    reason TEXT NOT NULL,
    UNIQUE (floor_plan_id, item, reason)
);

CREATE TABLE IF NOT EXISTS mask_color_evidence (
    floor_plan_id INTEGER NOT NULL REFERENCES floor_plans(image_id) ON DELETE CASCADE,
    candidate_name TEXT NOT NULL,
    rgb_red INTEGER NOT NULL,
    rgb_green INTEGER NOT NULL,
    rgb_blue INTEGER NOT NULL,
    mapping_status TEXT NOT NULL CHECK (mapping_status = 'CANDIDATE'),
    exact_rgb_pixels INTEGER NOT NULL,
    exact_rgb_percent REAL NOT NULL,
    lab_matched_pixels INTEGER NOT NULL,
    lab_matched_percent REAL NOT NULL,
    semantic_use TEXT NOT NULL,
    PRIMARY KEY (floor_plan_id, candidate_name)
);

CREATE INDEX IF NOT EXISTS idx_elements_plan_category
    ON elements(floor_plan_id, category);
CREATE INDEX IF NOT EXISTS idx_elements_category_status
    ON elements(category, semantic_status);
CREATE INDEX IF NOT EXISTS idx_elements_plan_confidence
    ON elements(floor_plan_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_uncertainties_plan
    ON uncertainties(floor_plan_id);
CREATE INDEX IF NOT EXISTS idx_color_evidence_candidate
    ON mask_color_evidence(candidate_name, mapping_status);
"""


def open_database(path: Path | str = DEFAULT_DATABASE_PATH, readonly: bool = False) -> sqlite3.Connection:
    database_path = Path(path).resolve()
    if readonly:
        if not database_path.is_file():
            raise FileNotFoundError(f"Database does not exist: {database_path}")
        uri = database_path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=10, factory=ClosingConnection)
        connection.execute("PRAGMA query_only = ON")
    else:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path, timeout=30, factory=ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(path: Path | str = DEFAULT_DATABASE_PATH) -> Path:
    database_path = Path(path).resolve()
    with open_database(database_path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute("PRAGMA user_version = 1")
    return database_path


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _validate_record(record: dict[str, Any], json_path: Path) -> None:
    required = {
        "image_id",
        "image_dimensions",
        "candidate_rooms",
        "candidate_walls",
        "candidate_doors",
        "candidate_windows",
        "mask_analysis",
        "uncertain_or_failed_detections",
        "input_files",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"{json_path} is missing required fields: {sorted(missing)}")
    if not isinstance(record["image_id"], int):
        raise ValueError(f"Invalid image ID in {json_path}")


def _category_rows(record: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for collection, category in (
        ("candidate_rooms", "room"),
        ("candidate_walls", "wall"),
        ("candidate_doors", "door"),
        ("candidate_windows", "window"),
    ):
        for detection in record.get(collection, []):
            yield category, detection


def import_extraction_record(
    connection: sqlite3.Connection,
    record: dict[str, Any],
    json_path: Path,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, int]:
    _validate_record(record, json_path)
    image_id = int(record["image_id"])
    files = record["input_files"]
    image_path = dataset_root / "X" / str(files["image"])
    mask_path = dataset_root / "Y" / str(files["mask"])
    annotated_path = annotated_dir / f"image_{image_id:03d}_annotated.png"
    dimensions = record["image_dimensions"]
    now = datetime.now(timezone.utc).isoformat()

    with connection:
        connection.execute(
            """INSERT INTO floor_plans (
                image_id, image_filename, mask_filename, source_image_path,
                source_mask_path, annotated_image_path, extraction_json_path,
                image_width, image_height, extraction_status, semantic_status,
                confidence_interpretation, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                image_filename=excluded.image_filename,
                mask_filename=excluded.mask_filename,
                source_image_path=excluded.source_image_path,
                source_mask_path=excluded.source_mask_path,
                annotated_image_path=excluded.annotated_image_path,
                extraction_json_path=excluded.extraction_json_path,
                image_width=excluded.image_width,
                image_height=excluded.image_height,
                extraction_status=excluded.extraction_status,
                semantic_status=excluded.semantic_status,
                confidence_interpretation=excluded.confidence_interpretation,
                processed_at=excluded.processed_at""",
            (
                image_id,
                str(files["image"]),
                str(files["mask"]),
                _relative_path(image_path, project_root),
                _relative_path(mask_path, project_root),
                _relative_path(annotated_path, project_root),
                _relative_path(json_path, project_root),
                int(dimensions["width"]),
                int(dimensions["height"]),
                "candidate_extraction",
                "CANDIDATE",
                str(record.get("confidence_interpretation", "heuristic, not calibrated")),
                now,
            ),
        )
        connection.execute("DELETE FROM elements WHERE floor_plan_id = ?", (image_id,))
        connection.execute("DELETE FROM uncertainties WHERE floor_plan_id = ?", (image_id,))
        connection.execute("DELETE FROM mask_color_evidence WHERE floor_plan_id = ?", (image_id,))

        element_count = 0
        for category, detection in _category_rows(record):
            box = detection["bounding_box"]
            connection.execute(
                """INSERT INTO elements (
                    detection_id, floor_plan_id, category, detection_type, semantic_status,
                    bbox_x, bbox_y, bbox_width, bbox_height, pixel_area, area_unit,
                    confidence, confidence_kind, detection_method, evidence_json, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pixels', ?, 'heuristic_not_calibrated', ?, ?, ?)""",
                (
                    str(detection["id"]),
                    image_id,
                    category,
                    str(detection["type"]),
                    str(detection.get("semantic_status", "CANDIDATE")),
                    int(box["x"]),
                    int(box["y"]),
                    int(box["width"]),
                    int(box["height"]),
                    int(detection.get("pixel_area", 0)),
                    float(detection["confidence"]),
                    str(detection["detection_method"]),
                    json.dumps(detection.get("evidence", []), ensure_ascii=False),
                    json.dumps(detection, ensure_ascii=False),
                ),
            )
            element_count += 1

        for uncertainty in record.get("uncertain_or_failed_detections", []):
            connection.execute(
                "INSERT OR IGNORE INTO uncertainties(floor_plan_id, item, reason) VALUES (?, ?, ?)",
                (image_id, str(uncertainty.get("item", "unknown")), str(uncertainty.get("reason", ""))),
            )

        color_count = 0
        for candidate in record.get("mask_analysis", {}).get("candidate_color_coverage", []):
            red, green, blue = (int(value) for value in candidate["rgb_candidate"])
            connection.execute(
                """INSERT INTO mask_color_evidence (
                    floor_plan_id, candidate_name, rgb_red, rgb_green, rgb_blue,
                    mapping_status, exact_rgb_pixels, exact_rgb_percent,
                    lab_matched_pixels, lab_matched_percent, semantic_use
                ) VALUES (?, ?, ?, ?, ?, 'CANDIDATE', ?, ?, ?, ?, ?)
                ON CONFLICT(floor_plan_id, candidate_name) DO UPDATE SET
                    rgb_red=excluded.rgb_red, rgb_green=excluded.rgb_green,
                    rgb_blue=excluded.rgb_blue, mapping_status='CANDIDATE',
                    exact_rgb_pixels=excluded.exact_rgb_pixels,
                    exact_rgb_percent=excluded.exact_rgb_percent,
                    lab_matched_pixels=excluded.lab_matched_pixels,
                    lab_matched_percent=excluded.lab_matched_percent,
                    semantic_use=excluded.semantic_use""",
                (
                    image_id,
                    str(candidate["name"]),
                    red,
                    green,
                    blue,
                    int(candidate["exact_rgb_pixels"]),
                    float(candidate["exact_rgb_percent"]),
                    int(candidate["lab_matched_pixels"]),
                    float(candidate["lab_matched_percent"]),
                    str(candidate.get("semantic_use", "supporting_evidence_only")),
                ),
            )
            color_count += 1
    return {"elements": element_count, "uncertainties": len(record.get("uncertain_or_failed_detections", [])), "colors": color_count}


def import_extraction_directory(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
    json_dir: Path = DEFAULT_JSON_DIR,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    initialize_database(database_path)
    json_files = sorted(json_dir.glob("image_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No extraction JSON files found in {json_dir}")
    report: dict[str, Any] = {"plans_imported": 0, "elements_imported": 0, "errors": []}
    with open_database(database_path) as connection:
        for json_path in json_files:
            try:
                with json_path.open(encoding="utf-8") as handle:
                    record = json.load(handle)
                counts = import_extraction_record(
                    connection,
                    record,
                    json_path,
                    dataset_root=dataset_root,
                    annotated_dir=annotated_dir,
                    project_root=project_root,
                )
                report["plans_imported"] += 1
                report["elements_imported"] += counts["elements"]
            except Exception as error:
                report["errors"].append(
                    {"file": json_path.name, "error": f"{type(error).__name__}: {error}"}
                )
    if report["errors"]:
        raise RuntimeError(f"Imported with {len(report['errors'])} errors: {report['errors'][:3]}")
    return report


def list_floor_plans(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """SELECT fp.*, COUNT(e.detection_id) AS element_count
        FROM floor_plans fp LEFT JOIN elements e ON e.floor_plan_id=fp.image_id
        GROUP BY fp.image_id ORDER BY fp.image_id"""
    ).fetchall()
    return [dict(row) for row in rows]


def get_floor_plan(connection: sqlite3.Connection, image_id: int) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM floor_plans WHERE image_id = ?", (image_id,)).fetchone()
    return dict(row) if row else None


def get_elements(
    connection: sqlite3.Connection,
    image_id: int,
    category: str | None = None,
) -> list[dict[str, Any]]:
    if category is None:
        rows = connection.execute(
            "SELECT * FROM elements WHERE floor_plan_id = ? ORDER BY category, bbox_y, bbox_x",
            (image_id,),
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM elements WHERE floor_plan_id = ? AND category = ? ORDER BY bbox_y, bbox_x",
            (image_id, category),
        ).fetchall()
    return [dict(row) for row in rows]


def get_uncertainties(connection: sqlite3.Connection, image_id: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT item, reason FROM uncertainties WHERE floor_plan_id = ? ORDER BY uncertainty_id",
        (image_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize and populate the SQLite database.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = initialize_database(args.database)
    if args.initialize_only:
        print(f"Initialized SQLite database: {path}")
        return
    result = import_extraction_directory(
        args.database,
        args.json_dir,
        args.dataset_root,
        DEFAULT_ANNOTATED_DIR,
        PROJECT_ROOT,
    )
    print(f"Database: {path}")
    print(f"Imported floor plans: {result['plans_imported']}")
    print(f"Imported candidate elements: {result['elements_imported']}")


if __name__ == "__main__":
    main()