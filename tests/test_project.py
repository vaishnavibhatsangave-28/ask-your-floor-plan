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

    def test_gemini_missing_key_is_explicit(self) -> None:
        result = GeminiAssistant(self.database_path, api_key="").ask(
            "How many candidate rooms are in plan 8?", 8
        )
        self.assertEqual(result["status"], "unconfigured")
        self.assertIn("GEMINI_API_KEY", result["answer"])

    def test_gemini_manual_function_call_uses_database_tool(self) -> None:
        class FakeModels:
            def __init__(self) -> None:
                self.calls = 0

            def generate_content(self, **_: object) -> SimpleNamespace:
                self.calls += 1
                if self.calls == 1:
                    call = SimpleNamespace(
                        name="get_floor_plan_summary",
                        args={"floor_plan_id": 8},
                    )
                    return SimpleNamespace(function_calls=[call], candidates=[], text="")
                return SimpleNamespace(function_calls=[], candidates=[], text="Database-backed candidate summary.")

        assistant = GeminiAssistant(self.database_path, api_key="unit-test-key")
        assistant.client = SimpleNamespace(models=FakeModels())
        result = assistant.ask("How many candidate rooms are in plan 8?", 8)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["tool_calls"][0]["tool"], "get_floor_plan_summary")
        self.assertTrue(result["tool_calls"][0]["sql"].startswith("SELECT"))
        self.assertEqual(result["tool_calls"][0]["rows"][0]["image_id"], 8)

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