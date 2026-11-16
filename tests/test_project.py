from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from database import (  # noqa: E402
    DEFAULT_ANNOTATED_DIR,
    DEFAULT_DATASET_ROOT,
    DEFAULT_JSON_DIR,
    PROJECT_ROOT as SOURCE_ROOT,
    import_extraction_directory,
    initialize_database,
    open_database,
)
from evaluation import run_evaluation  # noqa: E402
from extraction_pipeline import validate_extraction_record  # noqa: E402
from gemini_assistant import GeminiAssistant  # noqa: E402
from query_tools import execute_tool  # noqa: E402


class ProjectIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="ask-floor-plan-tests-")
        cls.database_path = Path(cls.temporary_directory.name) / "test.sqlite3"
        initialize_database(cls.database_path)
        result = import_extraction_directory(
            cls.database_path,
            DEFAULT_JSON_DIR,
            DEFAULT_DATASET_ROOT,
            DEFAULT_ANNOTATED_DIR,
            SOURCE_ROOT,
        )
        if result["plans_imported"] != 121 or result["errors"]:
            raise RuntimeError(f"Expected 121 source extraction records: {result}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_import_is_idempotent_and_preserves_uncertainty(self) -> None:
        import_extraction_directory(
            self.database_path,
            DEFAULT_JSON_DIR,
            DEFAULT_DATASET_ROOT,
            DEFAULT_ANNOTATED_DIR,
            SOURCE_ROOT,
        )
        with open_database(self.database_path, readonly=True) as connection:
            plans = connection.execute("SELECT COUNT(*) FROM floor_plans").fetchone()[0]
            elements = connection.execute("SELECT COUNT(*) FROM elements").fetchone()[0]
            uncertainty = connection.execute("SELECT COUNT(*) FROM uncertainties").fetchone()[0]
            candidate_only = connection.execute(
                "SELECT COUNT(*) FROM elements WHERE semantic_status != 'CANDIDATE'"
            ).fetchone()[0]
        expected_elements = 0
        for json_path in DEFAULT_JSON_DIR.glob("image_*.json"):
            record = json.loads(json_path.read_text(encoding="utf-8"))
            expected_elements += sum(
                len(record[key])
                for key in (
                    "candidate_rooms",
                    "candidate_walls",
                    "candidate_doors",
                    "candidate_windows",
                )
            )
        self.assertEqual(plans, len(list(DEFAULT_JSON_DIR.glob("image_*.json"))))
        self.assertEqual(elements, expected_elements)
        self.assertGreater(uncertainty, 0)
        self.assertEqual(candidate_only, 0)

    def test_all_121_extraction_json_records_validate(self) -> None:
        json_paths = sorted(DEFAULT_JSON_DIR.glob("image_*.json"))
        self.assertEqual(len(json_paths), 121)
        for image_id, json_path in enumerate(json_paths, start=1):
            record = json.loads(json_path.read_text(encoding="utf-8"))
            validate_extraction_record(
                record,
                image_id,
                f"{image_id}.png",
                f"{image_id} (2).png",
            )

    def test_readonly_connection_rejects_writes(self) -> None:
        with open_database(self.database_path, readonly=True) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM floor_plans")

    def test_parameterized_count_and_summary(self) -> None:
        count = execute_tool(
            "count_elements",
            {"floor_plan_id": 8, "category": "room"},
            str(self.database_path),
        )
        self.assertTrue(count["sql"].startswith("SELECT"))
        self.assertEqual(count["parameters"], [8, "room"])
        self.assertGreaterEqual(count["rows"][0]["candidate_count"], 0)

        summary = execute_tool(
            "get_floor_plan_summary", {"floor_plan_id": 61}, str(self.database_path)
        )
        self.assertEqual(summary["rows"][0]["image_id"], 61)
        self.assertIn("candidate_rooms", summary["rows"][0])
        self.assertIn("uncertainty_count", summary["rows"][0])

    def test_query_registry_rejects_raw_sql_and_invalid_category(self) -> None:
        with self.assertRaises(ValueError):
            execute_tool("run_sql", {"sql": "DROP TABLE elements"}, str(self.database_path))
        with self.assertRaises(ValueError):
            execute_tool(
                "count_elements",
                {"category": "room' OR 1=1--"},
                str(self.database_path),
            )

    def test_gemini_missing_key_is_explicit_for_open_questions(self) -> None:
        result = GeminiAssistant(self.database_path, api_key="").ask(
            "What do the candidate detections imply about circulation in plan 8?", 8
        )
        self.assertEqual(result["status"], "unconfigured")
        self.assertIn("GEMINI_API_KEY", result["answer"])

    def test_database_first_answers_count_question_without_gemini(self) -> None:
        assistant = GeminiAssistant(self.database_path, api_key="")
        result = assistant.ask(
            "How many candidate rooms, walls, doors, and windows are detected in floor plan 8?",
            8,
        )
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["source"], "database")
        self.assertEqual(result["tool_calls"][0]["tool"], "get_floor_plan_summary")
        row = result["tool_calls"][0]["rows"][0]
        self.assertEqual(int(row["candidate_rooms"]), 11)
        self.assertEqual(int(row["candidate_walls"]), 16)
        self.assertEqual(int(row["candidate_doors"]), 4)
        self.assertEqual(int(row["candidate_windows"]), 2)
        self.assertIn("11", result["answer"])
        self.assertIn("16", result["answer"])
        self.assertIn("4", result["answer"])
        self.assertIn("2", result["answer"])
        self.assertTrue(result["tool_calls"][0]["sql"].startswith("SELECT"))

    def test_database_first_count_and_compare_and_refusal(self) -> None:
        assistant = GeminiAssistant(self.database_path, api_key="")
        doors = assistant.ask("How many candidate doors are in plan 31?", 31)
        self.assertEqual(doors["status"], "answered")
        self.assertEqual(doors["tool_calls"][0]["tool"], "count_elements")
        door_count = int(doors["tool_calls"][0]["rows"][0]["candidate_count"])
        self.assertGreater(door_count, 0)
        self.assertIn(str(door_count), doors["answer"])

        comparison = assistant.ask("Compare candidate room counts in plans 8 and 31.")
        self.assertEqual(comparison["status"], "answered")
        self.assertEqual(comparison["tool_calls"][0]["tool"], "compare_room_counts")
        counts = {
            int(row["floor_plan_id"]): int(row["candidate_room_count"])
            for row in comparison["tool_calls"][0]["rows"]
        }
        self.assertEqual(counts[8], 11)
        self.assertIn(31, counts)
        self.assertIn("11", comparison["answer"])

        refused = assistant.ask("What is the total room area in square metres for plan 8?")
        self.assertEqual(refused["status"], "refused")
        self.assertIn("scale", refused["answer"].lower())

    def test_gemini_manual_function_call_uses_user_role_for_tool_response(self) -> None:
        captured: list[object] = []

        class FakeModels:
            def __init__(self) -> None:
                self.calls = 0

            def generate_content(self, **kwargs: object) -> SimpleNamespace:
                self.calls += 1
                captured.append(kwargs.get("contents"))
                if self.calls == 1:
                    call = SimpleNamespace(
                        name="find_adjacent_rooms",
                        args={"floor_plan_id": 8, "tolerance_pixels": 5},
                    )
                    return SimpleNamespace(function_calls=[call], candidates=[], text="")
                return SimpleNamespace(
                    function_calls=[],
                    candidates=[],
                    text="Geometry is a bounding-box estimate only.",
                )

        assistant = GeminiAssistant(self.database_path, api_key="unit-test-key")
        assistant.client = SimpleNamespace(models=FakeModels())
        result = assistant.ask(
            "What do the stored candidate detections imply about circulation in plan 8?",
            8,
        )
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["tool_calls"][0]["tool"], "find_adjacent_rooms")
        self.assertTrue(result["tool_calls"][0]["sql"].startswith("SELECT"))
        self.assertGreaterEqual(len(captured), 2)
        follow_up = captured[1]
        roles = [getattr(item, "role", None) for item in follow_up]
        self.assertNotIn("tool", roles)
        function_roles = [
            getattr(item, "role", None)
            for item in follow_up
            if getattr(item, "parts", None)
            and any(getattr(part, "function_response", None) for part in item.parts)
        ]
        self.assertTrue(function_roles)
        self.assertTrue(all(role == "user" for role in function_roles))

    def test_evaluation_contains_twenty_data_derived_cases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ask-floor-plan-eval-") as temporary:
            payload = run_evaluation(
                self.database_path,
                Path(temporary),
            )
        self.assertEqual(payload["summary"]["total_questions"], 20)
        self.assertEqual(payload["summary"]["correct_answers"], 15)
        self.assertEqual(payload["summary"]["correct_refusals"], 5)
        self.assertEqual(payload["summary"]["live_gemini_tests"], 0)
        self.assertTrue(all(result["status"] in {"passed", "correct_refusal"} for result in payload["results"]))

    def test_streamlit_pages_smoke(self) -> None:
        from streamlit.testing.v1 import AppTest

        app_test = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app_test.exception), 0)
        for page in ("Floor Plan Explorer", "Ask Your Floor Plan", "Evaluation"):
            app_test.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(app_test.exception), 0, page)


if __name__ == "__main__":
    unittest.main()