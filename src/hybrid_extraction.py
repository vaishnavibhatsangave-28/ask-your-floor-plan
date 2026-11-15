from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from class_mapping_validation import (
    CANDIDATE_CLASSES,
    UNKNOWN_RED_RGB,
    classify_mask,
    rgb_palette_lab,
)
from mask_analyzer import collect_images, read_rgb


DEFAULT_IMAGE_IDS = (8, 31, 61)
LAB_DISTANCE_LIMIT = 12.0
AMBIGUITY_MARGIN = 3.0
ROOM_MIN_AREA_RATIO = 0.005
ROOM_MAX_AREA_RATIO = 0.18
ROOM_MIN_SIDE_PIXELS = 45
ROOM_MIN_MASK_SUPPORT = 0.35
OPENING_MAX_AREA_RATIO = 0.008
ROOM_MIN_RECTANGULARITY = 0.35
MAX_WALL_SEGMENTS = 16
MAX_ANNOTATED_BY_CATEGORY = {"room": 10, "wall": 12, "door": 6, "window": 6}

ANNOTATION_COLORS_RGB = {
    "room": (0, 150, 255),
    "wall": (220, 40, 180),
    "door": (255, 210, 0),
    "window": (50, 210, 60),
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run conservative hybrid floor-plan extraction for selected numeric IDs."
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
        help="Folder for per-plan JSON and annotated images.",
    )
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_IMAGE_IDS),
        help="Numeric X/Y pair IDs to process; defaults to 8, 31, and 61.",
    )
    return parser.parse_args()


def load_image_pair(
    image_id: int,
    x_images: dict[int, Path],
    y_images: dict[int, Path],
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    if image_id not in x_images or image_id not in y_images:
        raise FileNotFoundError(f"No matching X/Y image-mask pair for numeric ID {image_id}")
    image_path = x_images[image_id]
    mask_path = y_images[image_id]
    image_rgb = read_rgb(image_path)
    mask_rgb = read_rgb(mask_path)
    if image_rgb.shape[:2] != mask_rgb.shape[:2]:
        raise ValueError(
            f"Pair {image_id} dimensions differ: {image_rgb.shape[:2]} vs {mask_rgb.shape[:2]}"
        )
    return image_rgb, mask_rgb, image_path, mask_path


def preprocess_image(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, dark_ink = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    edges = cv2.Canny(blurred, 50, 150, L2gradient=True)
    return {"gray": gray, "blurred": blurred, "dark_ink": dark_ink, "edges": edges}


def analyze_mask_support(mask_rgb: np.ndarray) -> dict[str, Any]:
    palette_lab = rgb_palette_lab()
    classification = classify_mask(
        mask_rgb,
        palette_lab,
        distance_limit=LAB_DISTANCE_LIMIT,
        ambiguity_margin=AMBIGUITY_MARGIN,
    )
    assignment = np.asarray(classification["assignment"], dtype=np.int16)
    total_pixels = int(mask_rgb.shape[0] * mask_rgb.shape[1])
    class_pixel_counts = np.asarray(classification["class_counts"], dtype=np.int64)
    exact_pixel_counts = np.asarray(classification["exact_counts"], dtype=np.int64)
    classes = []
    for index, candidate in enumerate(CANDIDATE_CLASSES):
        classes.append(
            {
                "name": candidate.name,
                "rgb_candidate": list(candidate.rgb),
                "mapping_status": "CANDIDATE",
                "exact_rgb_pixels": int(exact_pixel_counts[index]),
                "exact_rgb_percent": 100 * int(exact_pixel_counts[index]) / total_pixels,
                "lab_matched_pixels": int(class_pixel_counts[index]),
                "lab_matched_percent": 100 * int(class_pixel_counts[index]) / total_pixels,
                "semantic_use": "supporting_evidence_only",
            }
        )
    red_pixels = int(
        np.all(mask_rgb == np.asarray(UNKNOWN_RED_RGB, dtype=np.uint8), axis=2).sum()
    )
    return {
        "assignment": assignment,
        "class_summaries": classes,
        "colored_support": (np.max(mask_rgb, axis=2) > 24).astype(np.uint8) * 255,
        "red_exact_pixels": red_pixels,
        "ambiguous_pixels": int(classification["ambiguous_count"]),
        "unmatched_pixels": int(classification["unmatched_count"]),
    }


def get_mask_roi(colored_support: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(colored_support)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    padding = max(8, int(max(colored_support.shape) * 0.008))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(colored_support.shape[1], x + width + padding)
    y1 = min(colored_support.shape[0], y + height + padding)
    return x0, y0, x1 - x0, y1 - y0


def _bbox(x: int, y: int, width: int, height: int) -> dict[str, int]:
    return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}


def _component_color_evidence(
    assignment: np.ndarray,
    component: np.ndarray,
) -> list[dict[str, Any]]:
    values, counts = np.unique(assignment[component], return_counts=True)
    evidence = []
    for value, count in zip(values, counts):
        if value < 0 or value >= len(CANDIDATE_CLASSES):
            continue
        candidate = CANDIDATE_CLASSES[int(value)]
        evidence.append(
            {
                "candidate_name": candidate.name,
                "candidate_rgb": list(candidate.rgb),
                "pixels": int(count),
                "status": "CANDIDATE",
            }
        )
    return sorted(evidence, key=lambda row: row["pixels"], reverse=True)


def assess_confidence(
    *,
    rectangularity: float,
    light_pixel_fraction: float,
    edge_boundary_fraction: float,
    mask_support_fraction: float,
    size_score: float,
) -> float:
    score = (
        0.10
        + 0.14 * min(rectangularity, 1.0)
        + 0.12 * min(light_pixel_fraction, 1.0)
        + 0.16 * min(edge_boundary_fraction, 1.0)
        + 0.22 * min(mask_support_fraction, 1.0)
        + 0.10 * min(size_score, 1.0)
    )
    return round(min(score, 0.72), 3)


def detect_candidate_rooms(
    image_id: int,
    mask_analysis: dict[str, Any],
    preprocessing: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    support = np.asarray(mask_analysis["colored_support"], dtype=np.uint8)
    assignment = np.asarray(mask_analysis["assignment"], dtype=np.int16)
    roi = get_mask_roi(support)
    if roi is None:
        return []
    roi_x, roi_y, roi_width, roi_height = roi
    roi_slice = np.s_[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    dark_ink = np.asarray(preprocessing["dark_ink"][roi_slice], dtype=np.uint8)
    closed_ink = cv2.morphologyEx(
        dark_ink,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    enclosed_light_regions = cv2.bitwise_not(closed_ink)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        enclosed_light_regions, connectivity=8
    )
    roi_area = int(roi_width * roi_height)
    minimum_area = max(2400, int(roi_area * ROOM_MIN_AREA_RATIO))
    maximum_area = int(roi_area * ROOM_MAX_AREA_RATIO)
    unknown_index = _candidate_class_index("Unknown/background")
    assignment_roi = assignment[roi_slice]
    non_background_support = (assignment_roi >= 0) & (assignment_roi != unknown_index)
    candidates: list[dict[str, Any]] = []
    for label in range(1, component_count):
        local_x, local_y, width, height, area = (int(value) for value in stats[label])
        if (
            area < minimum_area
            or area > maximum_area
            or width < ROOM_MIN_SIDE_PIXELS
            or height < ROOM_MIN_SIDE_PIXELS
        ):
            continue
        aspect_ratio = max(width / height, height / width)
        rectangularity = area / float(width * height)
        if aspect_ratio > 8 or rectangularity < ROOM_MIN_RECTANGULARITY:
            continue

        component = labels == label
        border = cv2.morphologyEx(
            component.astype(np.uint8) * 255,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        ) > 0
        boundary_pixels = int(border.sum())
        edge_boundary_fraction = (
            float(
                np.count_nonzero(
                    (preprocessing["edges"][roi_slice] > 0) & border
                )
            )
            / boundary_pixels
            if boundary_pixels
            else 0.0
        )
        light_pixel_fraction = float(
            np.mean(preprocessing["gray"][roi_slice][component] > 170)
        )
        mapped = assignment_roi[component]
        mask_support_fraction = float(
            np.count_nonzero(non_background_support & component)
        ) / max(1, area)
        if mask_support_fraction < ROOM_MIN_MASK_SUPPORT:
            continue
        size_score = min(area / max(minimum_area * 4, 1), 1.0)
        confidence = assess_confidence(
            rectangularity=rectangularity,
            light_pixel_fraction=light_pixel_fraction,
            edge_boundary_fraction=edge_boundary_fraction,
            mask_support_fraction=min(mask_support_fraction, 1.0),
            size_score=size_score,
        )
        confidence = min(confidence, 0.68)
        x = roi_x + local_x
        y = roi_y + local_y
        candidates.append(
            {
                "id": f"{image_id}-room-region-{len(candidates) + 1:03d}",
                "type": "candidate_enclosed_region",
                "semantic_status": "CANDIDATE",
                "semantic_interpretation": "room_or_enclosed_space_candidate",
                "bounding_box": _bbox(x, y, width, height),
                "pixel_area": int(area),
                "confidence": confidence,
                "detection_method": "Otsu dark-ink threshold, 7x7 morphological close, enclosed-light connected components",
                "evidence": [
                    "Geometrically enclosed light region found inside the mask-derived building ROI.",
                    f"Original-image light-pixel fraction: {light_pixel_fraction:.3f}.",
                    f"Original-image Canny edge overlap on region boundary: {edge_boundary_fraction:.3f}.",
                    "Mask palette overlap is supporting evidence only; region is not confirmed as a room.",
                ],
                "shape_metrics": {
                    "rectangularity": round(rectangularity, 4),
                    "aspect_ratio": round(aspect_ratio, 4),
                    "light_pixel_fraction": round(light_pixel_fraction, 4),
                    "edge_boundary_fraction": round(edge_boundary_fraction, 4),
                },
                "mask_color_support": _component_color_evidence(assignment_roi, component),
            }
        )
    return candidates


def _candidate_class_index(name: str) -> int:
    return next(index for index, candidate in enumerate(CANDIDATE_CLASSES) if candidate.name == name)


def detect_candidate_openings(
    image_id: int,
    semantic_class: str,
    mask_analysis: dict[str, Any],
    preprocessing: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    assignment = np.asarray(mask_analysis["assignment"], dtype=np.int16)
    class_index = _candidate_class_index(semantic_class)
    candidate_pixels = (assignment == class_index).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_pixels, connectivity=8)
    total_pixels = int(candidate_pixels.size)
    wall_index = _candidate_class_index("Wall")
    wall_support = (assignment == wall_index).astype(np.uint8)
    wall_neighborhood = cv2.dilate(wall_support, np.ones((9, 9), dtype=np.uint8)) > 0

    detections: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < 16 or area > int(total_pixels * OPENING_MAX_AREA_RATIO):
            continue
        component = labels == label
        local_x0 = max(0, x - 5)
        local_y0 = max(0, y - 5)
        local_x1 = min(candidate_pixels.shape[1], x + width + 5)
        local_y1 = min(candidate_pixels.shape[0], y + height + 5)
        local_component = component[local_y0:local_y1, local_x0:local_x1]
        local_wall = wall_neighborhood[local_y0:local_y1, local_x0:local_x1]
        wall_adjacency = float(np.count_nonzero(local_component & local_wall)) / max(1, area)
        local_edges = preprocessing["edges"][local_y0:local_y1, local_x0:local_x1] > 0
        edge_fraction = float(np.count_nonzero(local_edges)) / max(1, local_edges.size)
        rectangularity = area / float(max(1, width * height))
        if wall_adjacency < 0.10 and edge_fraction < 0.025:
            continue

        size_score = 1.0 - min(area / max(total_pixels * OPENING_MAX_AREA_RATIO, 1), 1.0)
        confidence = assess_confidence(
            rectangularity=rectangularity,
            light_pixel_fraction=0.0,
            edge_boundary_fraction=min(edge_fraction * 10, 1.0),
            mask_support_fraction=min(wall_adjacency, 1.0),
            size_score=size_score,
        )
        is_door = semantic_class == "Door"
        detection_type = "candidate_door_marker" if is_door else "candidate_window_marker"
        detections.append(
            {
                "id": f"{image_id}-{semantic_class.lower()}-marker-{len(detections) + 1:03d}",
                "type": detection_type,
                "semantic_status": "CANDIDATE",
                "bounding_box": _bbox(x, y, width, height),
                "pixel_area": int(area),
                "confidence": confidence,
                "detection_method": "candidate-color connected component gated by nearby geometry",
                "evidence": [
                    f"Nearest palette support is the unconfirmed {semantic_class} candidate color.",
                    f"Nearby candidate-wall support fraction: {wall_adjacency:.3f}.",
                    f"Local Canny edge fraction: {edge_fraction:.3f}.",
                    "Marker is not confirmed as an architectural object; original SVG labels are unavailable.",
                ],
                "shape_metrics": {
                    "rectangularity": round(rectangularity, 4),
                    "wall_adjacency_fraction": round(wall_adjacency, 4),
                    "edge_fraction": round(edge_fraction, 4),
                },
            }
        )
    return detections


def detect_candidate_walls(
    image_id: int,
    preprocessing: dict[str, np.ndarray],
    mask_analysis: dict[str, Any],
    roi: tuple[int, int, int, int] | None,
) -> list[dict[str, Any]]:
    if roi is None:
        return []
    x0, y0, width, height = roi
    edge_roi = np.zeros_like(preprocessing["edges"])
    edge_roi[y0 : y0 + height, x0 : x0 + width] = preprocessing["edges"][
        y0 : y0 + height, x0 : x0 + width
    ]
    min_dimension = min(width, height)
    max_dimension = max(width, height)
    min_line_length = max(120, int(min_dimension * 0.11))
    max_line_gap = max(6, int(max_dimension * 0.003))
    hough_threshold = max(70, int(min_line_length * 0.72))
    lines = cv2.HoughLinesP(
        edge_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []

    assignment = np.asarray(mask_analysis["assignment"], dtype=np.int16)
    wall_index = _candidate_class_index("Wall")
    candidates: list[dict[str, Any]] = []
    line_segments = np.asarray(lines).reshape(-1, 4)
    ordered_lines = sorted(
        (tuple(int(value) for value in line) for line in line_segments),
        key=lambda line: (line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2,
        reverse=True,
    )
    accepted_segments: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in ordered_lines:
        length = float(np.hypot(x2 - x1, y2 - y1))
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        orientation = "horizontal" if angle <= 7 or angle >= 173 else None
        if 83 <= angle <= 97:
            orientation = "vertical"
        if orientation is None:
            continue
        line_coordinate = (y1 + y2) / 2 if orientation == "horizontal" else (x1 + x2) / 2
        line_interval = sorted((x1, x2)) if orientation == "horizontal" else sorted((y1, y2))
        duplicate = False
        for ax1, ay1, ax2, ay2 in accepted_segments:
            adx = ax2 - ax1
            ady = ay2 - ay1
            aangle = abs(float(np.degrees(np.arctan2(ady, adx))))
            aorientation = "horizontal" if aangle <= 7 or aangle >= 173 else None
            if 83 <= aangle <= 97:
                aorientation = "vertical"
            if aorientation != orientation:
                continue
            other_coordinate = (ay1 + ay2) / 2 if orientation == "horizontal" else (ax1 + ax2) / 2
            other_interval = sorted((ax1, ax2)) if orientation == "horizontal" else sorted((ay1, ay2))
            overlap = max(0, min(line_interval[1], other_interval[1]) - max(line_interval[0], other_interval[0]))
            shorter = max(1, min(line_interval[1] - line_interval[0], other_interval[1] - other_interval[0]))
            if abs(line_coordinate - other_coordinate) <= max(6, int(min_dimension * 0.005)) and overlap / shorter >= 0.55:
                duplicate = True
                break
        if duplicate:
            continue
        accepted_segments.append((x1, y1, x2, y2))

        line_mask = np.zeros_like(assignment, dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=5)
        line_pixels = line_mask > 0
        line_pixel_count = int(line_pixels.sum())
        if line_pixel_count == 0:
            continue
        dark_support = float(
            np.count_nonzero((preprocessing["dark_ink"] > 0) & line_pixels)
        ) / line_pixel_count
        if dark_support < 0.50:
            continue
        wall_color_support = float(np.count_nonzero((assignment == wall_index) & line_pixels)) / line_pixel_count
        length_score = min(length / max(min_line_length * 3, 1), 1.0)
        confidence = round(
            min(0.24 + 0.22 * dark_support + 0.18 * wall_color_support + 0.18 * length_score, 0.68),
            3,
        )
        bbox_x = min(x1, x2)
        bbox_y = min(y1, y2)
        bbox_width = abs(x2 - x1) + 1
        bbox_height = abs(y2 - y1) + 1
        candidates.append(
            {
                "id": f"{image_id}-wall-line-{len(candidates) + 1:03d}",
                "type": "candidate_wall_line",
                "semantic_status": "CANDIDATE",
                "bounding_box": _bbox(bbox_x, bbox_y, bbox_width, bbox_height),
                "line_segment": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "pixel_area": line_pixel_count,
                "confidence": confidence,
                "detection_method": "Canny edges plus probabilistic Hough line transform inside mask-support ROI",
                "evidence": [
                    f"Geometric dark-ink support along segment: {dark_support:.3f}.",
                    f"Unconfirmed candidate-wall palette overlap: {wall_color_support:.3f}.",
                    f"Segment length: {length:.1f} pixels.",
                    "A long edge may be a wall, dimension line, furniture edge, or drawing annotation.",
                ],
            }
        )
        if len(candidates) >= MAX_WALL_SEGMENTS:
            break
    return candidates


def assess_image_candidates(
    image_id: int,
    image_rgb: np.ndarray,
    mask_rgb: np.ndarray,
) -> dict[str, Any]:
    preprocessing = preprocess_image(image_rgb)
    mask_analysis = analyze_mask_support(mask_rgb)
    roi = get_mask_roi(np.asarray(mask_analysis["colored_support"]))
    rooms = detect_candidate_rooms(image_id, mask_analysis, preprocessing)
    walls = detect_candidate_walls(image_id, preprocessing, mask_analysis, roi)
    doors = detect_candidate_openings(image_id, "Door", mask_analysis, preprocessing)
    windows = detect_candidate_openings(image_id, "Window", mask_analysis, preprocessing)

    uncertain: list[dict[str, str]] = []
    if not rooms:
        uncertain.append(
            {
                "item": "candidate_rooms",
                "reason": "No mask-supported component passed the conservative geometric size/shape filters.",
            }
        )
    if not walls:
        uncertain.append(
            {
                "item": "candidate_walls",
                "reason": "No long Hough line segment passed the edge-based filters in the mask ROI.",
            }
        )
    if not doors:
        uncertain.append(
            {
                "item": "candidate_doors",
                "reason": "No door-color candidate marker also passed the nearby-edge/wall geometry gate.",
            }
        )
    if not windows:
        uncertain.append(
            {
                "item": "candidate_windows",
                "reason": "No window-color candidate marker also passed the nearby-edge/wall geometry gate.",
            }
        )
    uncertain.append(
        {
            "item": "semantic_classification",
            "reason": "All geometry and mask-supported outputs are candidates; semantic mappings and object classes are unverified.",
        }
    )
    if len(walls) >= MAX_WALL_SEGMENTS:
        uncertain.append(
            {
                "item": "candidate_walls",
                "reason": f"Wall segments reached the configured cap of {MAX_WALL_SEGMENTS}; remaining edges may be omitted.",
            }
        )
    if int(mask_analysis["red_exact_pixels"]) > 0:
        uncertain.append(
            {
                "item": "unassigned_red_color",
                "reason": "RGB (214, 70, 70) is recorded as unmatched; no semantic meaning is assigned.",
            }
        )

    detections = {
        "candidate_rooms": rooms,
        "candidate_walls": walls,
        "candidate_doors": doors,
        "candidate_windows": windows,
    }
    return {
        "image_id": image_id,
        "image_dimensions": {"width": int(image_rgb.shape[1]), "height": int(image_rgb.shape[0])},
        "candidate_status_policy": "All detections are CANDIDATE; no class mapping or extraction is confirmed.",
        "confidence_interpretation": "Heuristic evidence score in [0, 1], not a calibrated probability or accuracy estimate.",
        "candidate_rooms": rooms,
        "candidate_walls": walls,
        "candidate_doors": doors,
        "candidate_windows": windows,
        "mask_analysis": {
            "mapping_status": "candidate_only",
            "color_distance_method": "OpenCV 8-bit Lab nearest palette, Euclidean distance",
            "color_distance_limit": LAB_DISTANCE_LIMIT,
            "ambiguity_margin": AMBIGUITY_MARGIN,
            "candidate_color_coverage": mask_analysis["class_summaries"],
            "unassigned_red_rgb": list(UNKNOWN_RED_RGB),
            "unassigned_red_pixels": int(mask_analysis["red_exact_pixels"]),
            "ambiguous_pixels": int(mask_analysis["ambiguous_pixels"]),
            "unmatched_pixels": int(mask_analysis["unmatched_pixels"]),
            "mask_support_roi": _bbox(*roi) if roi is not None else None,
        },
        "processing": {
            "preprocessing": ["RGB to grayscale", "3x3 Gaussian blur", "Otsu dark-ink threshold", "Canny edge detection"],
            "geometric_methods": ["connected components", "shape and area filtering", "probabilistic Hough lines"],
            "parameters": {
                "room_min_area_ratio": ROOM_MIN_AREA_RATIO,
                "room_max_area_ratio": ROOM_MAX_AREA_RATIO,
                "opening_max_area_ratio": OPENING_MAX_AREA_RATIO,
                "max_wall_segments": MAX_WALL_SEGMENTS,
                "room_min_side_pixels": ROOM_MIN_SIDE_PIXELS,
                "room_min_nonbackground_mask_support": ROOM_MIN_MASK_SUPPORT,
            },
        },
        "uncertain_or_failed_detections": uncertain,
    }


def _draw_dashed_rectangle(
    image_bgr: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    dash_length: int = 12,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    for x in range(x1, x2, dash_length * 2):
        cv2.line(image_bgr, (x, y1), (min(x + dash_length, x2), y1), color, 2)
        cv2.line(image_bgr, (x, y2), (min(x + dash_length, x2), y2), color, 2)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(image_bgr, (x1, y), (x1, min(y + dash_length, y2)), color, 2)
        cv2.line(image_bgr, (x2, y), (x2, min(y + dash_length, y2)), color, 2)


def annotate_image(
    image_rgb: np.ndarray,
    result: dict[str, Any],
    output_path: Path,
    max_dimension: int = 2400,
) -> None:
    height, width = image_rgb.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1:
        display = cv2.resize(
            image_rgb,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display = image_rgb.copy()
    annotated = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

    for key, color_rgb in ANNOTATION_COLORS_RGB.items():
        detections = result[f"candidate_{key}s"]
        if key == "wall":
            detections = sorted(
                detections,
                key=lambda item: float(
                    np.hypot(
                        item["line_segment"]["x2"] - item["line_segment"]["x1"],
                        item["line_segment"]["y2"] - item["line_segment"]["y1"],
                    )
                ),
                reverse=True,
            )
        else:
            detections = sorted(
                detections,
                key=lambda item: float(item["confidence"]),
                reverse=True,
            )
        detections = detections[:MAX_ANNOTATED_BY_CATEGORY[key]]
        for detection in detections:
            box = detection["bounding_box"]
            x1 = int(round(box["x"] * scale))
            y1 = int(round(box["y"] * scale))
            x2 = int(round((box["x"] + box["width"]) * scale))
            y2 = int(round((box["y"] + box["height"]) * scale))
            color_bgr = tuple(reversed(color_rgb))
            if "line_segment" in detection:
                segment = detection["line_segment"]
                p1 = (int(round(segment["x1"] * scale)), int(round(segment["y1"] * scale)))
                p2 = (int(round(segment["x2"] * scale)), int(round(segment["y2"] * scale)))
                cv2.line(annotated, p1, p2, color_bgr, 2)
                label_x, label_y = p1
            else:
                _draw_dashed_rectangle(annotated, (x1, y1), (x2, y2), color_bgr)
                label_x, label_y = x1, y1
            if key != "wall" or detection is detections[0]:
                cv2.putText(
                    annotated,
                    f"{key[0].upper()}? {detection['confidence']:.2f}",
                    (max(0, label_x), max(18, label_y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color_bgr,
                    1,
                    cv2.LINE_AA,
                )

    legend = [
        "DASHED = CANDIDATE ONLY; no confirmed detections",
        "R? room region   W? wall line   D? door marker   N? window marker",
        "Mask palette is supporting evidence, not semantic ground truth",
    ]
    panel_width = min(annotated.shape[1] - 20, 960)
    panel_height = 92
    cv2.rectangle(annotated, (10, 10), (10 + panel_width, 10 + panel_height), (255, 255, 255), -1)
    cv2.rectangle(annotated, (10, 10), (10 + panel_width, 10 + panel_height), (40, 40, 40), 1)
    for index, line in enumerate(legend):
        cv2.putText(
            annotated,
            line,
            (22, 34 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), annotated)


def export_json(result: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def process_plan(
    image_id: int,
    x_images: dict[int, Path],
    y_images: dict[int, Path],
    json_dir: Path,
    visual_dir: Path,
) -> dict[str, Any]:
    image_rgb, mask_rgb, image_path, mask_path = load_image_pair(image_id, x_images, y_images)
    result = assess_image_candidates(image_id, image_rgb, mask_rgb)
    result["input_files"] = {"image": image_path.name, "mask": mask_path.name}
    export_json(result, json_dir / f"image_{image_id:03d}.json")
    annotate_image(image_rgb, result, visual_dir / f"image_{image_id:03d}_annotated.png")
    return result


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    json_dir = output_dir / "json"
    visual_dir = output_dir / "annotated"
    json_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    x_images = collect_images(dataset_root / "X", is_mask=False)
    y_images = collect_images(dataset_root / "Y", is_mask=True)
    results = []
    for image_id in dict.fromkeys(args.image_ids):
        results.append(process_plan(image_id, x_images, y_images, json_dir, visual_dir))

    print(f"Processed {len(results)} selected image-mask pairs: {', '.join(map(str, args.image_ids))}")
    print(f"JSON outputs: {json_dir}")
    print(f"Annotated images: {visual_dir}")
    for result in results:
        print(
            f"ID {result['image_id']}: rooms={len(result['candidate_rooms'])}, "
            f"walls={len(result['candidate_walls'])}, doors={len(result['candidate_doors'])}, "
            f"windows={len(result['candidate_windows'])}, "
            f"uncertainties={len(result['uncertain_or_failed_detections'])}"
        )


if __name__ == "__main__":
    main()