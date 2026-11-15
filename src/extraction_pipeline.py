from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from hybrid_extraction import annotate_image, load_image_pair, process_plan, read_rgb
from mask_analyzer import collect_images


REQUIRED_TOP_LEVEL_FIELDS = {
    "image_id",
    "image_dimensions",
    "candidate_rooms",
    "candidate_walls",
    "candidate_doors",
    "candidate_windows",
    "mask_analysis",
    "processing",
    "uncertain_or_failed_detections",
    "input_files",
}
ELEMENT_COLLECTIONS = (
    "candidate_rooms",
    "candidate_walls",
    "candidate_doors",
    "candidate_windows",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run or reuse hybrid extraction records for matched X/Y pairs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "dataset" / "raw" / "archive",
        help="Folder containing X/ and Y/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "extraction",
        help="Folder containing json/ and annotated/ subfolders.",
    )
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional numeric IDs to process; defaults to all matched pairs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run extraction and replace the selected JSON/annotated outputs.",
    )
    return parser.parse_args()


def validate_detection(detection: dict[str, Any], width: int, height: int) -> None:
    required = {
        "id",
        "type",
        "semantic_status",
        "bounding_box",
        "pixel_area",
        "confidence",
        "detection_method",
        "evidence",
    }
    missing = required - detection.keys()
    if missing:
        raise ValueError(f"Detection {detection.get('id', '<unknown>')} missing {sorted(missing)}")
    if detection["semantic_status"] not in {"CANDIDATE", "CONFIRMED", "UNKNOWN"}:
        raise ValueError(f"Unexpected semantic status: {detection['semantic_status']}")
    confidence = detection["confidence"]
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError(f"Invalid confidence for {detection['id']}: {confidence}")
    if int(detection["pixel_area"]) < 0:
        raise ValueError(f"Negative pixel area for {detection['id']}")
    box = detection["bounding_box"]
    x, y = int(box["x"]), int(box["y"])
    box_width, box_height = int(box["width"]), int(box["height"])
    if x < 0 or y < 0 or box_width < 0 or box_height < 0:
        raise ValueError(f"Invalid bounding box for {detection['id']}: {box}")
    if x + box_width > width or y + box_height > height:
        raise ValueError(f"Bounding box outside source dimensions for {detection['id']}: {box}")
    if not isinstance(detection["evidence"], list) or not detection["detection_method"]:
        raise ValueError(f"Missing evidence or detection method for {detection['id']}")


def validate_extraction_record(
    record: dict[str, Any],
    image_id: int,
    image_filename: str,
    mask_filename: str,
) -> None:
    missing = REQUIRED_TOP_LEVEL_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Extraction record missing fields: {sorted(missing)}")
    if record["image_id"] != image_id:
        raise ValueError(f"Record ID mismatch: expected {image_id}, found {record['image_id']}")
    if record["input_files"] != {"image": image_filename, "mask": mask_filename}:
        raise ValueError(f"Input filename mismatch for image ID {image_id}")
    dimensions = record["image_dimensions"]
    width, height = int(dimensions["width"]), int(dimensions["height"])
    for collection in ELEMENT_COLLECTIONS:
        if not isinstance(record[collection], list):
            raise ValueError(f"{collection} must be a list")
        for detection in record[collection]:
            validate_detection(detection, width, height)
    if not isinstance(record["uncertain_or_failed_detections"], list):
        raise ValueError("uncertain_or_failed_detections must be a list")


def load_valid_record(
    json_path: Path,
    image_id: int,
    image_filename: str,
    mask_filename: str,
) -> dict[str, Any]:
    with json_path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    validate_extraction_record(record, image_id, image_filename, mask_filename)
    return record


def run_pipeline(
    dataset_root: Path,
    output_dir: Path,
    image_ids: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    json_dir = output_dir / "json"
    annotated_dir = output_dir / "annotated"
    json_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    x_images = collect_images(dataset_root / "X", is_mask=False)
    y_images = collect_images(dataset_root / "Y", is_mask=True)
    missing_x = sorted(y_images.keys() - x_images.keys())
    missing_y = sorted(x_images.keys() - y_images.keys())
    if missing_x or missing_y:
        raise ValueError(f"Unmatched X/Y IDs. X missing={missing_x}; Y missing={missing_y}")

    selected_ids = sorted(set(image_ids)) if image_ids is not None else sorted(x_images)
    unknown_ids = sorted(set(selected_ids) - set(x_images))
    if unknown_ids:
        raise ValueError(f"Unknown requested image IDs: {unknown_ids}")

    processed_ids: list[int] = []
    reused_ids: list[int] = []
    errors: list[dict[str, Any]] = []
    for image_id in selected_ids:
        json_path = json_dir / f"image_{image_id:03d}.json"
        annotated_path = annotated_dir / f"image_{image_id:03d}_annotated.png"
        x_path = x_images[image_id]
        y_path = y_images[image_id]
        try:
            record: dict[str, Any] | None = None
            if json_path.exists() and not force:
                try:
                    record = load_valid_record(json_path, image_id, x_path.name, y_path.name)
                except (OSError, ValueError, json.JSONDecodeError):
                    record = None

            if record is None:
                image_rgb, mask_rgb, _, _ = load_image_pair(image_id, x_images, y_images)
                result = process_plan(image_id, x_images, y_images, json_dir, annotated_dir)
                result["input_files"] = {"image": x_path.name, "mask": y_path.name}
                validate_extraction_record(result, image_id, x_path.name, y_path.name)
                with json_path.open("w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
                    handle.write("\n")
                processed_ids.append(image_id)
            else:
                reused_ids.append(image_id)
                if not annotated_path.is_file() or cv2.imread(str(annotated_path)) is None:
                    image_rgb = read_rgb(x_path)
                    annotate_image(image_rgb, record, annotated_path)

            if (image_id + 1) % 10 == 0 or image_id == selected_ids[-1]:
                print(
                    f"Progress: ID {image_id}; {len(processed_ids)} newly processed, "
                    f"{len(reused_ids)} reused, {len(errors)} errors"
                )
        except Exception as error:
            errors.append({"image_id": image_id, "error": f"{type(error).__name__}: {error}"})
            print(f"ERROR ID {image_id}: {errors[-1]['error']}")

    manifest = {
        "started_or_completed_utc": datetime.now(timezone.utc).isoformat(),
        "requested_count": len(selected_ids),
        "newly_processed_count": len(processed_ids),
        "reused_count": len(reused_ids),
        "error_count": len(errors),
        "image_ids": selected_ids,
        "newly_processed_ids": processed_ids,
        "reused_ids": reused_ids,
        "unmatched_x_ids": missing_x,
        "unmatched_y_ids": missing_y,
        "errors": errors,
        "forced": force,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(
        f"Extraction complete: requested={len(selected_ids)}, processed={len(processed_ids)}, "
        f"reused={len(reused_ids)}, errors={len(errors)}"
    )
    if errors:
        raise RuntimeError(f"Extraction completed with {len(errors)} failed plans; see run_manifest.json")
    return manifest


def main() -> None:
    args = parse_args()
    run_pipeline(
        args.dataset_root.resolve(),
        args.output_dir.resolve(),
        image_ids=args.image_ids,
        force=args.force,
    )


if __name__ == "__main__":
    main()