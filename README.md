# Ask Your Floor Plan

Ask Your Floor Plan is a prototype for exploring floor-plan drawings, storing preliminary geometric detections, and answering database-backed questions about those records.

## Status

- 121 matched image/mask pairs are available locally under `dataset/raw/archive/X/` and `dataset/raw/archive/Y/`.
- The hybrid OpenCV extractor has generated JSON and annotated images for all 121 plans.
- All detected rooms, walls, doors, and windows are **candidates**, not verified objects. Confidence is a heuristic evidence score, not a calibrated probability.
- Mask color semantics are inconsistent across plans and are used only as supporting evidence. The red mask color remains unassigned.
- The 20-case evaluation checks SQLite query behavior and deterministic unsupported-question refusals. It does not evaluate live Gemini answer quality.

## Architecture

```text
X image + matching Y mask
        |
        v
src/hybrid_extraction.py -> outputs/extraction/{json,annotated}
        |
        v
src/database.py -> database/ask_your_floor_plan.sqlite3
        |
        +--> src/query_tools.py -> fixed, parameterized SELECT operations
        |                              |
        |                              v
        |                     src/gemini_assistant.py
        |                              |
        +------------------------------+--> app.py
                                       |
src/evaluation.py ---------------------+--> outputs/evaluation/
```

## Dataset and attribution

The local archive structurally matches Kaggle's [Floor Plan dataset by AsutoshPrad](https://www.kaggle.com/datasets/asutoshprad/floor-plan-dataset), which describes a customized CVC-FP dataset with SVG masks converted to PNG. The original CVC-FP page describes ground truth for architectural categories, but the original SVG archive could not be accessed for direct matching. The Kaggle page lists its license as unknown; check current licensing and attribution terms before redistribution. Do not commit the downloaded X/Y data unless its redistribution rights are clear.

## Requirements

Use Python 3.10+ and the project virtual environment. The existing `requirements.txt` pins OpenCV, NumPy, Streamlit, `google-genai`, and `python-dotenv` among the project's dependencies.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run extraction

The extraction runner reuses valid records and only processes missing or invalid outputs by default. It matches X/Y by numeric ID and preserves the downloader-added Y suffix.

```powershell
python src/extraction_pipeline.py
```

To rerun particular plans, or force regeneration:

```powershell
python src/extraction_pipeline.py --image-ids 8 31 61
python src/extraction_pipeline.py --force
```

JSON files are written to `outputs/extraction/json/`; annotated candidate overlays are written to `outputs/extraction/annotated/`. JSON bounding boxes and area values use source-image pixels. No real-world scale is inferred.

## Initialize and import SQLite

```powershell
python src/database.py --initialize-only
python src/database.py
```

The importer is idempotent by floor-plan and detection IDs; rerunning it replaces each plan's stored detections transactionally instead of duplicating them. The schema stores plan references, dimensions, candidate elements, pixel bounding boxes/areas, heuristic confidence, methods, evidence, candidate color support, and uncertainty notes.

## Configure Gemini

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` locally. Never commit `.env` or paste the key into source code. The current environment has no configured key, so live Gemini calls have not been tested.

```powershell
Copy-Item .env.example .env
# Edit .env locally and add your key.
```

The assistant exposes only registered database tools. Every operation uses fixed SQL with bound parameters over a SQLite connection opened in read-only mode. Model-generated SQL is never executed. Questions about square metres, room names/types, parking doors, materials, compass orientation, or color semantics are explicitly refused when the data does not support them.

Example supported questions:

- How many candidate rooms are in plan 8?
- What is the largest candidate region in pixel area?
- Which plans have the most candidate room regions?
- Which candidate room boxes are adjacent in plan 31?
- How many candidate windows are in plan 61?

## Evaluation

```powershell
python src/evaluation.py
```

The reproducible 20-question suite derives expected values from imported database rows, checks 15 SELECT-backed operations, and checks five unsupported-question refusal rules. Results are saved to `outputs/evaluation/evaluation_results.json` and `outputs/evaluation/evaluation_report.md`. No live LLM accuracy is claimed.

## Run the application

```powershell
streamlit run app.py
```

The app includes a dashboard, floor-plan explorer, database-backed Gemini chat, and evaluation view. The Explorer shows source and annotated images, candidate boxes, pixel areas, confidence, methods, and evidence. The chat view shows the database tool, SQL, parameters, supporting rows, and uncertainty.

## Extraction methodology

The prototype applies grayscale conversion, Gaussian blur, Otsu dark-ink thresholding, Canny edges, morphological closing, connected-component geometry, and probabilistic Hough lines. The mask contributes candidate color support only. No pretrained model is downloaded, and no room names or physical measurements are fabricated.

## Current database and evaluation results

The checked-in evaluation report is generated from current local extraction records. It evaluates database operations and refusal guards, not the semantic correctness of detections. See `outputs/validation/dataset_wide/mapping_reliability_audit.md` and `outputs/reports/dataset_class_mapping.md` for the preceding mask and mapping investigations.

## Limitations

- No authoritative SVG annotations or semantic ground truth were available for accuracy measurement.
- Candidate room regions can include text or fixtures; Hough lines can include dimensions, furniture, and drawing marks.
- Door/window proposals depend partly on an unverified palette and can be false positives or missing.
- Heuristic confidence is not calibrated and must not be interpreted as probability.
- Adjacency is estimated from bounding-box proximity, not room topology.
- Pixel areas and coordinates cannot be converted to physical dimensions without a reliable scale.
- Gemini is not configured in the current environment; set a local API key before using live chat.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The integration tests use existing extracted records, a temporary SQLite database, and a local fake Gemini function-call response. They do not make network requests or use fabricated floor-plan detections.