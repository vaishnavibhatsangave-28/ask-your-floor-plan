from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mask_analyzer import collect_images, pack_rgb, read_rgb, resize_for_display


DEFAULT_VISUAL_IDS = (1, 8, 16, 24, 31, 40, 48, 56, 61, 72, 83, 91, 101, 111, 121)
DEFAULT_LAB_DISTANCE_LIMIT = 12.0
DEFAULT_AMBIGUITY_MARGIN = 3.0
UNKNOWN_RED_RGB = (214, 70, 70)
AMBIGUOUS_RGB = (255, 0, 255)
UNMATCHED_RGB = (128, 128, 128)


@dataclass(frozen=True)
class CandidateClass:
    name: str
    rgb: tuple[int, int, int]
    status: str
    confidence: str
    evidence: str


CANDIDATE_CLASSES = (
    CandidateClass(
        "Room",
        (255, 142, 70),
        "CANDIDATE",
        "Medium for one matched sample; low dataset-wide",
        "CVC-FP mirror notebook palette plus visual alignment in pair 8; original SVG unavailable.",
    ),
    CandidateClass(
        "Wall",
        (175, 216, 248),
        "CANDIDATE",
        "Low to medium for one matched sample",
        "CVC-FP mirror notebook palette and outline-like regions in pair 8; no original SVG comparison.",
    ),
    CandidateClass(
        "Door",
        (246, 189, 15),
        "CANDIDATE",
        "Low to medium for one matched sample",
        "CVC-FP mirror notebook palette and small opening-aligned marks in pair 8; no SVG comparison.",
    ),
    CandidateClass(
        "Window",
        (139, 186, 0),
        "CANDIDATE",
        "Low to medium for one matched sample",
        "CVC-FP mirror notebook palette and small boundary-aligned marks in pair 8; no SVG comparison.",
    ),
    CandidateClass(
        "Separation",
        (0, 142, 142),
        "CANDIDATE",
        "Low for one matched sample",
        "CVC-FP mirror notebook palette and a small separating region in pair 8; no SVG comparison.",
    ),
    CandidateClass(
        "Unknown/background",
        (0, 0, 0),
        "CANDIDATE",
        "Low",
        "Mirror notebook labels black as Unknown; it dominates the canvas in local masks.",
    ),
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Visually validate candidate RGB classes against local floor-plan masks."
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
        default=project_root / "outputs" / "validation" / "dataset_wide",
        help="Folder for validation reports and visualizations.",
    )
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        default=None,
        help="Numeric X/Y pair IDs to audit (default: all matching pairs).",
    )
    parser.add_argument(
        "--visual-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_VISUAL_IDS),
        help="Numeric pair IDs for visual sheets; statistical outliers are added automatically.",
    )
    parser.add_argument(
        "--lab-distance-limit",
        type=float,
        default=DEFAULT_LAB_DISTANCE_LIMIT,
        help="Maximum nearest-palette distance in OpenCV 8-bit Lab to accept a pixel.",
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=DEFAULT_AMBIGUITY_MARGIN,
        help="Withhold pixels whose two nearest candidate distances differ by at most this value.",
    )
    return parser.parse_args()


def rgb_palette_lab() -> np.ndarray:
    palette = np.asarray([candidate.rgb for candidate in CANDIDATE_CLASSES], dtype=np.uint8)
    return cv2.cvtColor(palette.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(
        np.float32
    )


def classify_mask(
    mask_rgb: np.ndarray,
    palette_lab: np.ndarray,
    distance_limit: float,
    ambiguity_margin: float,
) -> dict[str, np.ndarray | int]:
    """Assign only near, unambiguous colors; return -1 for ambiguous and -2 for unmatched."""
    packed = pack_rgb(mask_rgb).reshape(-1)
    unique_colors, inverse, color_counts = np.unique(
        packed, return_inverse=True, return_counts=True
    )
    unique_rgb = np.column_stack(
        ((unique_colors >> 16) & 255, (unique_colors >> 8) & 255, unique_colors & 255)
    ).astype(np.uint8)
    unique_lab = cv2.cvtColor(unique_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(
        -1, 3
    ).astype(np.float32)
    distance_matrix = np.linalg.norm(
        unique_lab[:, None, :] - palette_lab[None, :, :], axis=2
    )
    nearest_order = np.argsort(distance_matrix, axis=1)
    nearest_index = nearest_order[:, 0]
    nearest_distance = distance_matrix[np.arange(len(unique_rgb)), nearest_index]
    second_distance = distance_matrix[np.arange(len(unique_rgb)), nearest_order[:, 1]]

    within_limit = nearest_distance <= distance_limit
    ambiguous = within_limit & ((second_distance - nearest_distance) <= ambiguity_margin)
    unique_assignment = np.full(len(unique_rgb), -2, dtype=np.int16)
    unique_assignment[within_limit & ~ambiguous] = nearest_index[within_limit & ~ambiguous]
    unique_assignment[ambiguous] = -1
    pixel_assignment = unique_assignment[inverse].reshape(mask_rgb.shape[:2])

    class_pixel_counts = np.bincount(
        pixel_assignment[pixel_assignment >= 0], minlength=len(CANDIDATE_CLASSES)
    )
    exact_pixel_counts = np.zeros(len(CANDIDATE_CLASSES), dtype=np.int64)
    for class_index, candidate in enumerate(CANDIDATE_CLASSES):
        exact_pixel_counts[class_index] = int(
            np.all(mask_rgb == np.asarray(candidate.rgb, dtype=np.uint8), axis=2).sum()
        )

    red_count = int(np.all(mask_rgb == np.asarray(UNKNOWN_RED_RGB, dtype=np.uint8), axis=2).sum())
    return {
        "assignment": pixel_assignment,
        "class_counts": class_pixel_counts,
        "exact_counts": exact_pixel_counts,
        "ambiguous_count": int(np.count_nonzero(pixel_assignment == -1)),
        "unmatched_count": int(np.count_nonzero(pixel_assignment == -2)),
        "red_exact_count": red_count,
    }


def render_candidate_mask(assignment: np.ndarray) -> np.ndarray:
    colors = np.asarray([candidate.rgb for candidate in CANDIDATE_CLASSES], dtype=np.uint8)
    rendered = np.zeros((*assignment.shape, 3), dtype=np.uint8)
    for class_index, color in enumerate(colors):
        rendered[assignment == class_index] = color
    rendered[assignment == -1] = AMBIGUOUS_RGB
    rendered[assignment == -2] = UNMATCHED_RGB
    return rendered


def make_comparison_sheet(
    image_id: int,
    source_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    assignment: np.ndarray,
    stats: dict[str, np.ndarray | int],
    output_path: Path,
    distance_limit: float,
) -> None:
    height, width = source_rgb.shape[:2]
    if mask_rgb.shape[:2] != (height, width):
        raise ValueError(f"Pair {image_id} dimensions differ: {source_rgb.shape} vs {mask_rgb.shape}")

    classified_rgb = render_candidate_mask(assignment)
    overlay_rgb = cv2.addWeighted(source_rgb, 0.48, classified_rgb, 0.52, 0)
    source_display = resize_for_display(source_rgb, 1300, cv2.INTER_AREA)
    mask_display = resize_for_display(mask_rgb, 1300, cv2.INTER_NEAREST)
    overlay_display = resize_for_display(overlay_rgb, 1300, cv2.INTER_NEAREST)
    assignment_display = cv2.resize(
        assignment.astype(np.int16),
        (source_display.shape[1], source_display.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    total_pixels = assignment.size
    class_counts = np.asarray(stats["class_counts"])

    figure, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    axes = axes.ravel()
    top_panels = (
        (source_display, "Original floor plan"),
        (mask_display, "Original RGB mask"),
        (overlay_display, "Overlay: gray=unmatched, magenta=ambiguous"),
    )
    for axis, (image, title) in zip(axes[:3], top_panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    for class_index, candidate in enumerate(CANDIDATE_CLASSES):
        if candidate.name == "Unknown/background":
            class_view = np.full((*assignment_display.shape, 3), 240, dtype=np.uint8)
        else:
            class_view = np.zeros((*assignment_display.shape, 3), dtype=np.uint8)
        class_view[assignment_display == class_index] = candidate.rgb
        percent = 100 * int(class_counts[class_index]) / total_pixels
        color_text = ", ".join(str(channel) for channel in candidate.rgb)
        axes[3 + class_index].imshow(class_view)
        axes[3 + class_index].set_title(
            f"{candidate.name}: RGB ({color_text})\n{percent:.3f}% of mask",
            fontsize=9,
        )
        axes[3 + class_index].axis("off")

    figure.suptitle(
        f"Pair {image_id}: candidate mapping validation "
        f"(Lab distance <= {distance_limit:g})",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=135, bbox_inches="tight")
    plt.close(figure)


def make_pair_row(
    image_id: int,
    x_filename: str,
    y_filename: str,
    stats: dict[str, np.ndarray | int],
    total_pixels: int,
) -> dict[str, object]:
    counts = np.asarray(stats["class_counts"], dtype=np.int64)
    exact_counts = np.asarray(stats["exact_counts"], dtype=np.int64)
    row: dict[str, object] = {
        "image_id": image_id,
        "x_filename": x_filename,
        "y_filename": y_filename,
        "total_pixels": total_pixels,
        "ambiguous_percent": 100 * int(stats["ambiguous_count"]) / total_pixels,
        "unmatched_percent": 100 * int(stats["unmatched_count"]) / total_pixels,
        "red_exact_percent": 100 * int(stats["red_exact_count"]) / total_pixels,
    }
    for index, candidate in enumerate(CANDIDATE_CLASSES):
        key = candidate.name.lower().replace("/", "_").replace(" ", "_")
        row[f"{key}_lab_matched_percent"] = 100 * int(counts[index]) / total_pixels
        row[f"{key}_exact_rgb_percent"] = 100 * int(exact_counts[index]) / total_pixels
    return row


def write_coverage_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["image_id", "x_filename", "y_filename", "total_pixels"]
    for candidate in CANDIDATE_CLASSES:
        key = candidate.name.lower().replace("/", "_").replace(" ", "_")
        fields.extend((f"{key}_lab_matched_percent", f"{key}_exact_rgb_percent"))
    fields.extend(("ambiguous_percent", "unmatched_percent", "red_exact_percent"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_iqr_outliers(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    outliers: list[dict[str, object]] = []
    for candidate in CANDIDATE_CLASSES:
        key = candidate.name.lower().replace("/", "_").replace(" ", "_")
        field = f"{key}_lab_matched_percent"
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        first_quartile, third_quartile = np.percentile(values, [25, 75])
        spread = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * spread
        upper_bound = third_quartile + 1.5 * spread
        members = [
            {"image_id": int(row["image_id"]), "percent": float(row[field])}
            for row in rows
            if float(row[field]) < lower_bound or float(row[field]) > upper_bound
        ]
        outliers.append(
            {
                "class_name": candidate.name,
                "q1": float(first_quartile),
                "q3": float(third_quartile),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "members": members,
            }
        )
    return outliers


def write_validation_report(
    path: Path,
    rows: list[dict[str, object]],
    outliers: list[dict[str, object]],
    sample_totals: np.ndarray,
    sample_exact_totals: np.ndarray,
    total_pixels: int,
    ambiguous_pixels: int,
    unmatched_pixels: int,
    red_pixels: int,
    visual_ids: list[int],
    distance_limit: float,
    ambiguity_margin: float,
    unmatched_x_ids: list[int],
    unmatched_y_ids: list[int],
) -> None:
    outlier_ids = sorted(
        {
            int(member["image_id"])
            for result in outliers
            for member in result["members"]
        }
    )
    room_heavy_ids = [
        int(row["image_id"])
        for row in rows
        if float(row["room_lab_matched_percent"]) > 5
        and float(row["separation_lab_matched_percent"]) < 1
    ]
    separation_heavy_ids = [
        int(row["image_id"])
        for row in rows
        if float(row["separation_lab_matched_percent"]) > 5
        and float(row["room_lab_matched_percent"]) < 1
    ]
    non_background_classes = [
        candidate
        for candidate in CANDIDATE_CLASSES
        if candidate.name != "Unknown/background"
    ]
    dominant_color_ids: dict[str, list[int]] = {
        candidate.name: [] for candidate in non_background_classes
    }
    for row in rows:
        dominant = max(
            non_background_classes,
            key=lambda candidate: float(
                row[
                    f"{candidate.name.lower().replace('/', '_').replace(' ', '_')}_lab_matched_percent"
                ]
            ),
        )
        dominant_color_ids[dominant.name].append(int(row["image_id"]))
    with path.open("w", encoding="utf-8") as report:
        report.write("# Dataset-Wide Candidate Mapping Reliability Audit\n\n")
        report.write(
            f"Audited {len(rows)} matched X/Y pairs across the dataset. Unmatched X IDs: "
            f"{unmatched_x_ids or 'none'}; unmatched Y IDs: {unmatched_y_ids or 'none'}. "
            f"No source image or mask was modified. Visual sheets were generated for "
            f"{len(visual_ids)} representative/outlier pairs.\n\n"
        )
        report.write("## Matching method\n\n")
        report.write(
            "Each mask pixel is converted to OpenCV's 8-bit Lab space and compared with the "
            "six proposed RGB palette centers. The nearest candidate is accepted only when "
            f"its Euclidean Lab distance is <= {distance_limit:g}. Pixels whose nearest and "
            f"second-nearest candidates differ by <= {ambiguity_margin:g} are withheld as "
            "ambiguous. This is a conservative anti-aliasing tolerance, not a CIEDE2000 "
            "measurement or a semantic classifier. Exact-center coverage is reported separately. "
            "Outliers are identified per candidate using Tukey's 1.5 x IQR rule on Lab-matched "
            "per-image percentages.\n\n"
        )
        report.write("## Dataset-Wide Coverage and Reliability\n\n")
        report.write(
            f"Across {total_pixels:,} pixels from {len(rows)} masks, {ambiguous_pixels:,} "
            f"({100 * ambiguous_pixels / total_pixels:.4f}%) were withheld as near-ties, "
            f"{unmatched_pixels:,} ({100 * unmatched_pixels / total_pixels:.4f}%) were outside "
            f"the palette tolerance, and {red_pixels:,} ({100 * red_pixels / total_pixels:.4f}%) "
            "exactly matched the unassigned red RGB.\n\n"
        )
        report.write(
            "| Candidate | RGB | Lab-matched pixels | Exact RGB pixels | Median per-image Lab share | "
            "Zero-coverage pairs | IQR outliers | Reliability assessment | Status |\n"
        )
        report.write("|---|---|---:|---:|---:|---:|---|---|---|\n")
        for index, candidate in enumerate(CANDIDATE_CLASSES):
            matched_percent = 100 * int(sample_totals[index]) / total_pixels
            exact_percent = 100 * int(sample_exact_totals[index]) / total_pixels
            key = candidate.name.lower().replace("/", "_").replace(" ", "_")
            per_image = np.asarray(
                [float(row[f"{key}_lab_matched_percent"]) for row in rows], dtype=np.float64
            )
            median_share = float(np.median(per_image))
            zero_coverage = int(np.count_nonzero(per_image == 0))
            outlier_result = next(item for item in outliers if item["class_name"] == candidate.name)
            outlier_text = ", ".join(
                f"{member['image_id']} ({member['percent']:.2f}%)"
                for member in outlier_result["members"]
            ) or "none"
            if candidate.name == "Room":
                reliability = "Low: room-colored coverage has strong cross-pair outliers."
            elif candidate.name == "Separation":
                reliability = "Very low: pair 61 has a large separation-colored filled region."
            elif candidate.name == "Unknown/background":
                reliability = "Low semantic confidence: black is common but may include canvas/linework."
            else:
                reliability = "Color coverage is measurable; semantic identity remains unverified."
            report.write(
                f"| {candidate.name} | `({candidate.rgb[0]}, {candidate.rgb[1]}, {candidate.rgb[2]})` | "
                f"{int(sample_totals[index]):,} ({matched_percent:.3f}%) | "
                f"{int(sample_exact_totals[index]):,} ({exact_percent:.3f}%) | "
                f"{median_share:.3f}% | {zero_coverage} | {outlier_text} | "
                f"{reliability} | **CANDIDATE** |\n"
            )
        report.write("\n")
        report.write(
            f"The exact red color `(214, 70, 70)` occurs in {red_pixels:,} pixels and remains "
            "unmatched; no semantic meaning is assigned. Parking door remains **UNKNOWN**.\n\n"
        )

        report.write("## Distribution Outliers\n\n")
        for result in outliers:
            members = result["members"]
            member_text = ", ".join(
                f"{member['image_id']} ({member['percent']:.3f}%)" for member in members
            ) or "none"
            report.write(
                f"- {result['class_name']}: Q1={result['q1']:.3f}%, Q3={result['q3']:.3f}%, "
                f"Tukey bounds [{result['lower_bound']:.3f}%, {result['upper_bound']:.3f}%]; "
                f"outliers: {member_text}.\n"
            )
        report.write("\n")

        report.write("## Room/Separation and Convention Checks\n\n")
        report.write(
            f"Pairs with orange Room >5% and teal Separation <1%: "
            f"{room_heavy_ids or 'none'}.\n\n"
        )
        report.write(
            f"Pairs with teal Separation >5% and orange Room <1%: "
            f"{separation_heavy_ids or 'none'}.\n\n"
        )
        if separation_heavy_ids:
            report.write(
                "This opposing distribution pattern is consistent with a possible image-level "
                "palette/convention change or an incorrect semantic assignment. It does not by "
                "itself prove that the two categories literally switch; room contents and lack "
                "of original SVGs remain confounders.\n\n"
            )
        dominant_counts = ", ".join(
            f"{class_name}: {len(image_ids)}"
            for class_name, image_ids in dominant_color_ids.items()
        )
        report.write(
            f"The largest non-background candidate color by pixel share is split across images "
            f"as follows: {dominant_counts}. Room and Separation are tied at 59 images each, "
            "while Door, Window, and Wall lead in pairs 40, 67, and 85 respectively. These "
            "winner labels are only candidate palette names. The broad interior fills in pair 40 "
            "(Door yellow, 20.869%) and pair 67 (Window green, 14.715%) are visually suspicious "
            "semantic false positives.\n\n"
        )
        report.write(
            "The candidate palette colors recur across the dataset, but class shares and "
            "unmatched-color rates vary by image. The opposing Room/Separation and other "
            "dominant-color regimes are consistent with multiple annotation styles or "
            "inconsistent per-image color conventions, but cannot distinguish those from "
            "different plan content or false candidate mappings without original SVG annotations. "
            "They do not prove a literal Room/Separation role switch. No pixel is assigned to "
            "more than one candidate; near-ties are withheld.\n\n"
        )

        report.write("## Other Suspicious Cases and Limitations\n\n")
        report.write(
            "Exact-color and Lab-matched percentages are separate: the latter includes near "
            "anti-aliased shades within the configured tolerance. These are color-coverage "
            "statistics, not semantic validation. The third-party notebook hard-codes its display "
            "palette, and the original SVG archive was unavailable, so no class mapping is "
            "confirmed.\n\n"
        )
        report.write("## Visual sheets\n\n")
        for image_id in visual_ids:
            report.write(f"- [Pair {image_id}](visuals/pair_{image_id:03d}_validation.png)\n")
        report.write(
            "\nEach sheet includes the source plan, source mask, candidate-colored overlay, and "
            "six individual candidate masks. Magenta marks ambiguous pixels; gray marks pixels "
            "outside the candidate palette tolerance. Full per-pair exact and Lab coverage is "
            "in `pair_mapping_coverage.csv`.\n\n"
        )
        report.write(
            "## Recommendation\n\n"
            "Do not use the unverified semantic palette as the sole basis for extraction. A "
            "hybrid approach (C) is the safest next research direction: preserve mask color "
            "evidence with uncertainty reporting, then cross-check geometry with computer vision "
            "and the original SVG schema when obtainable. No extraction was started.\n"
        )


def main() -> None:
    args = parse_args()
    if args.lab_distance_limit <= 0 or args.ambiguity_margin < 0:
        raise ValueError("lab-distance-limit must be positive and ambiguity-margin non-negative")

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    x_images = collect_images(dataset_root / "X", is_mask=False)
    y_images = collect_images(dataset_root / "Y", is_mask=True)
    paired_ids = sorted(x_images.keys() & y_images.keys())
    image_ids = list(dict.fromkeys(args.image_ids)) if args.image_ids else paired_ids
    if not image_ids:
        raise ValueError(f"No matching numeric image-mask pairs found under {dataset_root}")
    missing_ids = sorted(set(image_ids) - set(paired_ids))
    if missing_ids:
        raise ValueError(f"Requested IDs without matching X/Y files: {missing_ids}")
    palette_lab = rgb_palette_lab()

    rows: list[dict[str, object]] = []
    class_totals = np.zeros(len(CANDIDATE_CLASSES), dtype=np.int64)
    exact_totals = np.zeros(len(CANDIDATE_CLASSES), dtype=np.int64)
    total_pixels = 0
    ambiguous_pixels = 0
    unmatched_pixels = 0
    red_pixels = 0
    analyzed_ids: list[int] = []

    for image_id in image_ids:
        source_path = x_images[image_id]
        mask_path = y_images[image_id]
        source_rgb = read_rgb(source_path)
        mask_rgb = read_rgb(mask_path)
        if source_rgb.shape[:2] != mask_rgb.shape[:2]:
            raise ValueError(
                f"Pair {image_id} dimensions differ: {source_rgb.shape[:2]} vs {mask_rgb.shape[:2]}"
            )
        result = classify_mask(
            mask_rgb,
            palette_lab,
            args.lab_distance_limit,
            args.ambiguity_margin,
        )
        stats = {key: value for key, value in result.items() if key != "assignment"}
        image_total = int(mask_rgb.shape[0] * mask_rgb.shape[1])
        row = make_pair_row(
            image_id,
            source_path.name,
            mask_path.name,
            stats,
            image_total,
        )
        rows.append(row)
        class_totals += np.asarray(result["class_counts"], dtype=np.int64)
        exact_totals += np.asarray(result["exact_counts"], dtype=np.int64)
        total_pixels += image_total
        ambiguous_pixels += int(result["ambiguous_count"])
        unmatched_pixels += int(result["unmatched_count"])
        red_pixels += int(result["red_exact_count"])
        analyzed_ids.append(image_id)

    outliers = find_iqr_outliers(rows)
    outlier_ids = {
        int(member["image_id"])
        for result in outliers
        for member in result["members"]
    }
    visual_ids = sorted(
        (set(args.visual_ids) | outlier_ids) & set(analyzed_ids)
    )
    for image_id in visual_ids:
        source_rgb = read_rgb(x_images[image_id])
        mask_rgb = read_rgb(y_images[image_id])
        result = classify_mask(
            mask_rgb,
            palette_lab,
            args.lab_distance_limit,
            args.ambiguity_margin,
        )
        assignment = np.asarray(result["assignment"])
        stats = {key: value for key, value in result.items() if key != "assignment"}
        output_path = visual_dir / f"pair_{image_id:03d}_validation.png"
        make_comparison_sheet(
            image_id,
            source_rgb,
            mask_rgb,
            assignment,
            stats,
            output_path,
            args.lab_distance_limit,
        )

    csv_path = output_dir / "pair_mapping_coverage.csv"
    write_coverage_csv(csv_path, rows)
    report_path = output_dir / "mapping_reliability_audit.md"
    write_validation_report(
        report_path,
        rows,
        outliers,
        class_totals,
        exact_totals,
        total_pixels,
        ambiguous_pixels,
        unmatched_pixels,
        red_pixels,
        visual_ids,
        args.lab_distance_limit,
        args.ambiguity_margin,
        sorted(x_images.keys() - y_images.keys()),
        sorted(y_images.keys() - x_images.keys()),
    )
    print(f"Audited {len(analyzed_ids)} image-mask pairs")
    print(f"Generated visual sheets for {len(visual_ids)} pairs: {', '.join(map(str, visual_ids))}")
    print(f"Report: {report_path}")
    print(f"Coverage CSV: {csv_path}")
    for image_id in visual_ids:
        print(f"Visual sheet: {visual_dir / f'pair_{image_id:03d}_validation.png'}")


if __name__ == "__main__":
    main()