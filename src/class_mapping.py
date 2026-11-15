from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mask_analyzer import collect_images, exact_color_counts, pack_rgb, read_rgb, resize_for_display


DEFAULT_REPRESENTATIVE_IDS = (1, 31, 61, 91, 121)
DEFAULT_GROUP_COUNT = 7
DEFAULT_SAMPLE_BUDGET = 1_000_000
RANDOM_SEED = 2026


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build visual evidence and a cautious color-to-class mapping report."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "dataset" / "raw" / "archive",
        help="Folder containing X/ and Y/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "outputs",
        help="Output directory.",
    )
    parser.add_argument(
        "--representative-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPRESENTATIVE_IDS),
        help="Numeric pair IDs to visualize.",
    )
    parser.add_argument(
        "--sample-budget",
        type=int,
        default=DEFAULT_SAMPLE_BUDGET,
        help="Maximum area-weighted pixels used to fit Lab palette centers.",
    )
    return parser.parse_args()


def rgb_to_lab(rgb_values: np.ndarray) -> np.ndarray:
    pixels = np.asarray(rgb_values, dtype=np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def nearest_palette_indices(rgb_values: np.ndarray, centers_lab: np.ndarray) -> np.ndarray:
    lab_values = rgb_to_lab(rgb_values)
    centers = centers_lab.astype(np.float32)
    distances = ((lab_values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1)


def fit_palette(
    pairs: dict[int, tuple[Path, Path]], sample_budget: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    areas: dict[int, int] = {}
    for image_id, (_, mask_path) in pairs.items():
        mask = read_rgb(mask_path)
        areas[image_id] = int(mask.shape[0] * mask.shape[1])

    total_area = sum(areas.values())
    sample_sizes = {
        image_id: max(1, round(sample_budget * area / total_area))
        for image_id, area in areas.items()
    }
    rng = np.random.default_rng(RANDOM_SEED)
    samples: list[np.ndarray] = []
    all_colors: Counter[int] = Counter()
    for image_id, (_, mask_path) in pairs.items():
        mask = read_rgb(mask_path)
        all_colors.update(exact_color_counts(mask))
        pixels = mask.reshape(-1, 3)
        sample_count = min(sample_sizes[image_id], len(pixels))
        indices = rng.choice(len(pixels), size=sample_count, replace=False)
        samples.append(pixels[indices])

    sample_rgb = np.concatenate(samples, axis=0)
    sample_lab = rgb_to_lab(sample_rgb)
    cv2.setRNGSeed(RANDOM_SEED)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers_lab = cv2.kmeans(
        sample_lab,
        DEFAULT_GROUP_COUNT,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()
    sample_counts = np.bincount(labels, minlength=DEFAULT_GROUP_COUNT)
    palette_rgb = np.zeros((DEFAULT_GROUP_COUNT, 3), dtype=np.uint8)
    for group_id in range(DEFAULT_GROUP_COUNT):
        palette_rgb[group_id] = np.rint(sample_rgb[labels == group_id].mean(axis=0)).astype(
            np.uint8
        )

    order = np.argsort(-sample_counts)
    centers_lab = centers_lab[order]
    palette_rgb = palette_rgb[order]

    packed_colors = np.fromiter(all_colors.keys(), dtype=np.uint32)
    color_rgb = np.column_stack(
        ((packed_colors >> 16) & 255, (packed_colors >> 8) & 255, packed_colors & 255)
    ).astype(np.uint8)
    color_groups = nearest_palette_indices(color_rgb, centers_lab)
    color_counts = np.fromiter(all_colors.values(), dtype=np.int64)
    pixel_counts = np.bincount(
        color_groups, weights=color_counts, minlength=DEFAULT_GROUP_COUNT
    ).round().astype(np.int64)
    return centers_lab, palette_rgb, pixel_counts, len(sample_rgb)


def group_mask(rgb: np.ndarray, centers_lab: np.ndarray) -> np.ndarray:
    packed = pack_rgb(rgb)
    unique_colors, inverse = np.unique(packed, return_inverse=True)
    unique_rgb = np.column_stack(
        ((unique_colors >> 16) & 255, (unique_colors >> 8) & 255, unique_colors & 255)
    ).astype(np.uint8)
    unique_groups = nearest_palette_indices(unique_rgb, centers_lab).astype(np.uint8)
    return unique_groups[inverse].reshape(rgb.shape[:2])


def per_pair_counts(
    mask_rgb: np.ndarray, centers_lab: np.ndarray, group_count: int
) -> np.ndarray:
    color_counts = exact_color_counts(mask_rgb)
    packed_colors = np.fromiter(color_counts.keys(), dtype=np.uint32)
    color_rgb = np.column_stack(
        ((packed_colors >> 16) & 255, (packed_colors >> 8) & 255, packed_colors & 255)
    ).astype(np.uint8)
    color_groups = nearest_palette_indices(color_rgb, centers_lab)
    frequencies = np.fromiter(color_counts.values(), dtype=np.int64)
    return np.bincount(color_groups, weights=frequencies, minlength=group_count).round().astype(
        np.int64
    )


def save_palette_figure(palette_rgb: np.ndarray, pixel_counts: np.ndarray, path: Path) -> None:
    total = int(pixel_counts.sum())
    figure, axis = plt.subplots(figsize=(13, 3.2), constrained_layout=True)
    axis.set_xlim(0, len(palette_rgb))
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xticks(np.arange(len(palette_rgb)) + 0.5)
    axis.set_xticklabels(
        [
            f"RGB {tuple(int(c) for c in color)}\n{count / total:.3%}"
            for color, count in zip(palette_rgb, pixel_counts)
        ],
        fontsize=8,
    )
    for index, color in enumerate(palette_rgb):
        axis.add_patch(plt.Rectangle((index, 0), 1, 1, color=color.astype(float) / 255.0))
    axis.set_title("Seven dominant Lab color groups (no semantic labels)")
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def make_pair_visualization(
    image_id: int,
    source_path: Path,
    mask_path: Path,
    centers_lab: np.ndarray,
    palette_rgb: np.ndarray,
    output_path: Path,
) -> np.ndarray:
    source = read_rgb(source_path)
    mask_rgb = read_rgb(mask_path)
    groups = group_mask(mask_rgb, centers_lab)
    counts = np.bincount(groups.ravel(), minlength=len(palette_rgb))
    shares = counts / counts.sum()

    source_display = resize_for_display(source, 1500, cv2.INTER_AREA)
    mask_display = resize_for_display(mask_rgb, 1500, cv2.INTER_NEAREST)
    group_display = resize_for_display(groups, 1500, cv2.INTER_NEAREST)
    colorized = palette_rgb[group_display]
    blended = cv2.addWeighted(source_display, 0.5, colorized, 0.5, 0)

    figure, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    axes = axes.ravel()
    panels = (
        (source_display, "Original floor plan"),
        (mask_display, "Original RGB mask"),
        (blended, "Mask groups overlaid on plan"),
    )
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    axes[3].axis("off")

    gray = cv2.cvtColor(source_display, cv2.COLOR_RGB2GRAY)
    dimmed_source = np.repeat((gray.astype(np.float32) * 0.25).astype(np.uint8)[..., None], 3, axis=2)
    for group_id, axis in enumerate(axes[4:11]):
        highlighted = dimmed_source.copy()
        selected = group_display == group_id
        highlighted[selected] = palette_rgb[group_id]
        color = tuple(int(channel) for channel in palette_rgb[group_id])
        axis.imshow(highlighted)
        axis.set_title(
            f"Group {group_id + 1}: RGB {color}\n{shares[group_id]:.2%} of pair",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes[11:]:
        axis.axis("off")

    figure.suptitle(
        f"Pair {image_id}: location evidence for seven dominant colors", fontsize=14
    )
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return counts


def write_report(
    report_path: Path,
    visual_dir: Path,
    palette_rgb: np.ndarray,
    pixel_counts: np.ndarray,
    pair_rows: list[dict[str, object]],
    representative_ids: list[int],
    sample_pixels: int,
) -> None:
    total_pixels = int(pixel_counts.sum())
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# Dataset Class Mapping Investigation\n\n")
        report.write("## Dataset source and documentation\n\n")
        report.write(
            "The local archive matches Kaggle's [Floor Plan dataset by AsutoshPrad]"
            "(https://www.kaggle.com/datasets/asutoshprad/floor-plan-dataset): the page "
            "identifies it as an enhanced CVC FPP dataset, lists X and Y, and reports 242 "
            "files. The local archive contains 121 X images and 121 Y masks (242 files), "
            "with matching numeric IDs and dimensions. This is a strong structural identity "
            "match, not a byte-for-byte provenance check.\n\n"
        )
        report.write(
            "The Kaggle uploader describes the targets as SVG masks converted to PNG with a "
            "custom script and acknowledges the original CVC FPP creators without naming "
            "them. Kaggle metadata dates the upload to 2024-10-28 and lists the license as "
            "**Unknown**. Its description does not define "
            "a class legend, RGB-to-class mapping, or annotation schema. The local dataset "
            "contains no non-image metadata or annotation files.\n\n"
        )
        report.write(
            "The associated Kaggle notebook [Floor Plan dataset]"
            "(https://www.kaggle.com/code/liubovcherepanova/notebookf9859148d5) identifies "
            "the same Kaggle input but provides no accessible class legend in its page "
            "metadata. No unrelated floor-plan datasets were used as mapping evidence.\n\n"
        )

        report.write("## Method and representative visual inspection\n\n")
        report.write(
            f"The seven color groups were fit by OpenCV K-means in CIE Lab space using a "
            f"deterministic, pixel-area-weighted sample of {sample_pixels:,} pixels "
            f"(seed {RANDOM_SEED}). Every exact RGB value found in all Y masks was assigned "
            "to its nearest fitted Lab center to calculate the table's pixel frequencies. "
            "That assignment is exhaustive given the fitted centers; group boundaries and "
            "representative colors remain approximate because of clustering and anti-aliasing.\n\n"
        )
        report.write(
            "Each visual sheet shows the original plan, original mask, a color-group overlay, "
            "and seven location maps over a dimmed plan. Percentages are measured per mask; "
            "the panels are evidence of geometry and color location only, not semantic proof.\n\n"
        )
        report.write(
            "A notable cross-pair observation is that large filled interior regions appear "
            "teal in pairs 1 and 61 but orange in pairs 31, 91, and 121. Thus a dominant "
            "filled-region appearance does not have a stable global RGB identity in these "
            "examples; the images may use varying palettes or annotation conventions. This "
            "prevents a reliable cross-dataset color-to-class assignment from visual evidence.\n\n"
        )
        for image_id in representative_ids:
            visual_path = visual_dir / f"pair_{image_id:03d}_class_evidence.png"
            if visual_path.exists():
                report.write(
                    f"- Pair {image_id}: [visual evidence](../visualizations/class_mapping/"
                    f"{visual_path.name})\n"
                )
        report.write("\n### Per-pair group coverage\n\n")
        report.write("| Pair | " + " | ".join(f"G{i}" for i in range(1, 8)) + " |\n")
        report.write("|---:|" + "---:|" * 7 + "\n")
        for row in pair_rows:
            shares = row["shares"]
            report.write(
                f"| {row['image_id']} | "
                + " | ".join(f"{share:.2%}" for share in shares)
                + " |\n"
            )
        report.write("\n### Dataset-wide groups and class assessment\n\n")
        report.write(
            "| Group | Representative RGB | Pixel frequency | Visual observations | "
            "Possible interpretation | Evidence source | Confidence | Mapping status |\n"
        )
        report.write(
            "|---:|---|---:|---|---|---|---|---|\n"
        )
        group_evidence = (
            (
                "Near-black fills most of the canvas outside colored regions; dark plan "
                "detail also remains in some areas.",
                "Unpainted background or retained linework; not an architectural object label.",
                "Low; visual only",
                "CANDIDATE",
            ),
            (
                "Broad filled regions in pairs 1 and 61; almost absent in pairs 31, 91, and 121.",
                "A filled-region role is plausible, but its RGB identity is not stable across pairs.",
                "Low for region role; none for class",
                "UNKNOWN",
            ),
            (
                "Broad filled regions in pairs 31, 91, and 121, but only small marks in pair 1.",
                "A filled-region role is plausible, but its RGB identity is not stable across pairs.",
                "Low for region role; none for class",
                "UNKNOWN",
            ),
            (
                "Mostly thin, light outline-like strokes around or within filled regions.",
                "Boundary or outline annotation is possible; not confirmed as walls or another class.",
                "Low; visual only",
                "CANDIDATE",
            ),
            (
                "Short, block-like marks distributed along plan edges and internal locations.",
                "Unidentified marker or annotation; no architectural class is supported.",
                "None for semantic identity",
                "UNKNOWN",
            ),
            (
                "Small isolated blocks or short marks, often near geometry boundaries.",
                "Unidentified marker or annotation; no architectural class is supported.",
                "None for semantic identity",
                "UNKNOWN",
            ),
            (
                "Sparse red line segments inside some plans; absent or negligible in some pairs.",
                "Internal line/mark annotation is possible; no object identity is supported.",
                "Low; visual only",
                "CANDIDATE",
            ),
        )
        for group_id, (color, count) in enumerate(zip(palette_rgb, pixel_counts)):
            rgb_tuple = tuple(int(channel) for channel in color)
            percentage = 100 * count / total_pixels
            observation, interpretation, confidence, status = group_evidence[group_id]
            report.write(
                f"| G{group_id + 1} | `({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]})` | "
                f"{int(count):,} px ({percentage:.4f}%) | {observation} | {interpretation} | "
                "Local paired X/Y PNGs; official Kaggle description confirms rasterized SVG "
                f"masks but supplies no legend | {confidence} | **{status}** |\n"
            )
        report.write("\n")
        report.write(
            "**Confirmed mappings: none.** No color can be reliably identified as a wall, "
            "door, window, room, or other architectural class from the available sources. "
            "The black group is only a low-confidence background candidate; do not treat it "
            "as confirmed. The seven clusters are palette groupings, not evidence that the "
            "dataset has seven semantic classes.\n\n"
        )
        report.write("## Palette visualization\n\n")
        report.write("![Seven grouped colors](../visualizations/class_mapping/dominant_palette.png)\n\n")
        report.write("## Limitations and next step\n\n")
        report.write(
            "The Kaggle page does not identify the original CVC FPP authors or expose the "
            "SVG annotation schema, and its license is unknown. A visual association is not "
            "authoritative documentation. Before implementing semantic mapping or extraction, "
            "obtain the original CVC FPP annotation guide/legend or inspect the original SVG "
            "layers and verify their IDs against the Kaggle-rendered PNGs. Request permission "
            "or clarification from the Kaggle uploader if the original schema is unavailable. "
            "Keep room, wall, door, and window extraction paused pending that verification.\n"
        )


def main() -> None:
    args = parse_args()
    if args.sample_budget < DEFAULT_GROUP_COUNT:
        raise ValueError(f"sample-budget must be at least {DEFAULT_GROUP_COUNT}")

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    report_dir = output_root / "reports"
    visual_dir = output_root / "visualizations" / "class_mapping"
    report_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    x_images = collect_images(dataset_root / "X", is_mask=False)
    y_images = collect_images(dataset_root / "Y", is_mask=True)
    shared_ids = sorted(x_images.keys() & y_images.keys())
    pairs = {
        image_id: (x_images[image_id], y_images[image_id])
        for image_id in shared_ids
    }
    if not pairs:
        raise ValueError(f"No numeric X/Y image pairs found under {dataset_root}")

    centers_lab, palette_rgb, pixel_counts, sample_pixels = fit_palette(
        pairs, args.sample_budget
    )
    selected_ids = list(dict.fromkeys(args.representative_ids))
    pair_rows: list[dict[str, object]] = []
    for image_id in selected_ids:
        if image_id not in pairs:
            continue
        source_path, mask_path = pairs[image_id]
        mask_rgb = read_rgb(mask_path)
        counts = per_pair_counts(mask_rgb, centers_lab, len(palette_rgb))
        output_path = visual_dir / f"pair_{image_id:03d}_class_evidence.png"
        make_pair_visualization(
            image_id,
            source_path,
            mask_path,
            centers_lab,
            palette_rgb,
            output_path,
        )
        pair_rows.append(
            {
                "image_id": image_id,
                "shares": counts / counts.sum(),
            }
        )

    palette_path = visual_dir / "dominant_palette.png"
    save_palette_figure(palette_rgb, pixel_counts, palette_path)

    csv_path = report_dir / "dataset_color_groups.csv"
    total_pixels = int(pixel_counts.sum())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "representative_rgb", "pixel_count", "percent"])
        for group_id, (color, count) in enumerate(zip(palette_rgb, pixel_counts), start=1):
            writer.writerow(
                [
                    group_id,
                    ",".join(map(str, (int(channel) for channel in color))),
                    int(count),
                    f"{100 * count / total_pixels:.8f}",
                ]
            )

    report_path = report_dir / "dataset_class_mapping.md"
    write_report(
        report_path,
        visual_dir,
        palette_rgb,
        pixel_counts,
        pair_rows,
        [int(row["image_id"]) for row in pair_rows],
        sample_pixels,
    )
    print(f"Analyzed {len(pairs)} image-mask pairs; representative pairs: {len(pair_rows)}")
    print(f"Report: {report_path}")
    print(f"Color groups: {csv_path}")
    print(f"Palette: {palette_path}")
    for row in pair_rows:
        print(f"Visual evidence: {visual_dir / f'pair_{int(row['image_id']):03d}_class_evidence.png'}")


if __name__ == "__main__":
    main()