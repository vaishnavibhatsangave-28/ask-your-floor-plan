from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_REPRESENTATIVE_IDS = (1, 31, 61, 91, 121)
DEFAULT_CLUSTER_COUNT = 8
DEFAULT_SAMPLE_BUDGET = 1_000_000
COLOR_CHUNK_ROWS = 128
PAIR_PATTERN = re.compile(r"^(\d+)(?: \(2\))?$")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Inspect paired floor-plan RGB masks.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "dataset" / "raw" / "archive",
        help="Folder containing X/ and Y/ (default: dataset/raw/archive).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "outputs",
        help="Output directory (default: outputs).",
    )
    parser.add_argument(
        "--representative-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPRESENTATIVE_IDS),
        help="Numeric pair IDs to visualize (default: 1 31 61 91 121).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=DEFAULT_CLUSTER_COUNT,
        help="Number of Lab-space palette groups (default: 8).",
    )
    parser.add_argument(
        "--sample-budget",
        type=int,
        default=DEFAULT_SAMPLE_BUDGET,
        help="Maximum dataset pixels sampled for clustering (default: 1000000).",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    return parser.parse_args()


def collect_images(folder: Path, is_mask: bool) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        stem = path.stem
        if is_mask:
            stem = stem.removesuffix(" (2)")
        match = re.fullmatch(r"\d+", stem)
        if match is None:
            continue
        image_id = int(stem)
        if image_id in images:
            raise ValueError(f"Duplicate image ID {image_id} in {folder}")
        images[image_id] = path
    return images


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encoded_format_counts(images: dict[int, Path]) -> Counter[str]:
    formats: Counter[str] = Counter()
    signatures = {
        b"\x89PNG\r\n\x1a\n": "PNG",
        b"\xff\xd8\xff": "JPEG",
    }
    for path in images.values():
        with path.open("rb") as handle:
            header = handle.read(8)
        image_format = next(
            (name for signature, name in signatures.items() if header.startswith(signature)),
            "unknown",
        )
        formats[image_format] += 1
    return formats


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.uint32, copy=False)
    return (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]


def unpack_rgb(value: int) -> tuple[int, int, int]:
    return (value >> 16) & 255, (value >> 8) & 255, value & 255


def exact_color_counts(rgb: np.ndarray) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row_start in range(0, rgb.shape[0], COLOR_CHUNK_ROWS):
        packed = pack_rgb(rgb[row_start : row_start + COLOR_CHUNK_ROWS])
        values, frequencies = np.unique(packed, return_counts=True)
        counts.update({int(value): int(count) for value, count in zip(values, frequencies)})
    return counts


def resize_for_display(rgb: np.ndarray, max_dimension: int, interpolation: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale == 1.0:
        return rgb
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, size, interpolation=interpolation)


def cluster_mask(rgb: np.ndarray, centers_lab: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    flat_lab = lab.reshape(-1, 3)
    result = np.empty((flat_lab.shape[0], 3), dtype=np.uint8)
    centers = centers_lab.astype(np.float32)
    for start in range(0, flat_lab.shape[0], 250_000):
        pixels = flat_lab[start : start + 250_000].astype(np.float32)
        distances = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        result[start : start + len(pixels)] = palette_rgb[np.argmin(distances, axis=1)]
    return result.reshape(rgb.shape)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_visualizations(
    pairs: dict[int, tuple[Path, Path]],
    representative_ids: list[int],
    centers_lab: np.ndarray,
    palette_rgb: np.ndarray,
    cluster_shares: np.ndarray,
    visual_dir: Path,
) -> list[Path]:
    visual_paths: list[Path] = []
    for image_id in representative_ids:
        if image_id not in pairs:
            continue
        source_path, mask_path = pairs[image_id]
        source = resize_for_display(read_rgb(source_path), 1600, cv2.INTER_AREA)
        mask = resize_for_display(read_rgb(mask_path), 1600, cv2.INTER_NEAREST)
        grouped = cluster_mask(mask, centers_lab, palette_rgb)

        figure, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
        panels = (
            (source, f"X/{source_path.name}: floor plan"),
            (mask, f"Y/{mask_path.name}: RGB mask"),
            (grouped, f"Y colors grouped in Lab (K={len(palette_rgb)})"),
        )
        for axis, (image, title) in zip(axes, panels):
            axis.imshow(image)
            axis.set_title(title, fontsize=11)
            axis.axis("off")
        figure.suptitle(f"Pair {image_id}: input, target, grouped target", fontsize=14)
        output_path = visual_dir / f"pair_{image_id:03d}_comparison.png"
        figure.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        visual_paths.append(output_path)

    figure, axis = plt.subplots(figsize=(12, 3.2), constrained_layout=True)
    axis.set_xlim(0, len(palette_rgb))
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xticks(np.arange(len(palette_rgb)) + 0.5)
    labels = [
        f"RGB {tuple(int(channel) for channel in color)}\n{share:.4%} sample"
        for color, share in zip(palette_rgb, cluster_shares)
    ]
    axis.set_xticklabels(labels, fontsize=8)
    for index, color in enumerate(palette_rgb):
        axis.add_patch(
            plt.Rectangle((index, 0), 1, 1, color=color.astype(float) / 255.0)
        )
    axis.set_title("Dataset-wide dominant palette (Lab clusters; no semantic labels)")
    palette_path = visual_dir / "dataset_dominant_palette.png"
    figure.savefig(palette_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    visual_paths.append(palette_path)
    return visual_paths


def main() -> None:
    args = parse_args()
    if args.clusters < 1 or args.sample_budget < args.clusters:
        raise ValueError("clusters must be positive and sample-budget must be >= clusters")

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    report_dir = output_root / "reports"
    visual_dir = output_root / "visualizations"
    report_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    x_images = collect_images(dataset_root / "X", is_mask=False)
    y_images = collect_images(dataset_root / "Y", is_mask=True)
    x_formats = encoded_format_counts(x_images)
    y_formats = encoded_format_counts(y_images)
    shared_ids = sorted(x_images.keys() & y_images.keys())
    if not shared_ids:
        raise ValueError(f"No numeric X/Y image pairs found under {dataset_root}")

    pairs = {image_id: (x_images[image_id], y_images[image_id]) for image_id in shared_ids}
    dimensions: Counter[tuple[int, int]] = Counter()
    pair_dimension_matches = 0
    mask_areas: dict[int, int] = {}
    for image_id, (source_path, mask_path) in pairs.items():
        source = read_rgb(source_path)
        mask = read_rgb(mask_path)
        dimensions[(source.shape[1], source.shape[0])] += 1
        pair_dimension_matches += int(source.shape[:2] == mask.shape[:2])
        mask_areas[image_id] = int(mask.shape[0] * mask.shape[1])

    total_area = sum(mask_areas.values())
    sample_sizes = {
        image_id: max(1, round(args.sample_budget * area / total_area))
        for image_id, area in mask_areas.items()
    }
    rng = np.random.default_rng(args.seed)
    exact_global: Counter[int] = Counter()
    per_mask_top_rows: list[dict[str, object]] = []
    samples: list[np.ndarray] = []

    for image_id, (_, mask_path) in pairs.items():
        mask = read_rgb(mask_path)
        counts = exact_color_counts(mask)
        exact_global.update(counts)
        pixels = mask.reshape(-1, 3)
        sample_count = min(sample_sizes[image_id], pixels.shape[0])
        indices = rng.choice(pixels.shape[0], size=sample_count, replace=False)
        samples.append(pixels[indices])
        total_pixels = pixels.shape[0]
        for rank, (packed, count) in enumerate(counts.most_common(10), start=1):
            red, green, blue = unpack_rgb(packed)
            per_mask_top_rows.append(
                {
                    "image_id": image_id,
                    "mask_file": mask_path.name,
                    "rank": rank,
                    "red": red,
                    "green": green,
                    "blue": blue,
                    "rgb": f"{red},{green},{blue}",
                    "pixel_count": count,
                    "percent_of_mask": f"{100 * count / total_pixels:.8f}",
                }
            )

    sample_rgb = np.concatenate(samples, axis=0)
    sample_lab = cv2.cvtColor(sample_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
    sample_lab = sample_lab.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(args.seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers_lab = cv2.kmeans(
        sample_lab,
        args.clusters,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()
    cluster_counts = np.bincount(labels, minlength=args.clusters)
    cluster_shares = cluster_counts / cluster_counts.sum()
    palette_rgb = np.zeros((args.clusters, 3), dtype=np.uint8)
    for cluster_id in range(args.clusters):
        members = sample_rgb[labels == cluster_id]
        palette_rgb[cluster_id] = np.rint(members.mean(axis=0)).astype(np.uint8)
    order = np.argsort(-cluster_counts)
    centers_lab = centers_lab[order]
    palette_rgb = palette_rgb[order]
    cluster_counts = cluster_counts[order]
    cluster_shares = cluster_shares[order]

    per_mask_csv = report_dir / "mask_top_rgb_colors.csv"
    write_csv(
        per_mask_csv,
        [
            "image_id",
            "mask_file",
            "rank",
            "red",
            "green",
            "blue",
            "rgb",
            "pixel_count",
            "percent_of_mask",
        ],
        per_mask_top_rows,
    )

    global_color_rows = []
    all_mask_pixels = sum(mask_areas.values())
    for rank, (packed, count) in enumerate(exact_global.most_common(50), start=1):
        red, green, blue = unpack_rgb(packed)
        global_color_rows.append(
            {
                "rank": rank,
                "red": red,
                "green": green,
                "blue": blue,
                "rgb": f"{red},{green},{blue}",
                "pixel_count": count,
                "percent_of_all_mask_pixels": f"{100 * count / all_mask_pixels:.8f}",
            }
        )
    global_csv = report_dir / "dataset_top_rgb_colors.csv"
    write_csv(
        global_csv,
        ["rank", "red", "green", "blue", "rgb", "pixel_count", "percent_of_all_mask_pixels"],
        global_color_rows,
    )

    cluster_rows = []
    for rank, (color, count, share) in enumerate(
        zip(palette_rgb, cluster_counts, cluster_shares), start=1
    ):
        rgb_tuple = tuple(int(channel) for channel in color)
        cluster_rows.append(
            {
                "rank": rank,
                "representative_rgb": ",".join(map(str, rgb_tuple)),
                "sample_pixel_count": int(count),
                "sample_percent": f"{100 * share:.6f}",
            }
        )
    cluster_csv = report_dir / "dataset_lab_palette.csv"
    write_csv(
        cluster_csv,
        ["rank", "representative_rgb", "sample_pixel_count", "sample_percent"],
        cluster_rows,
    )

    selected_ids = list(dict.fromkeys(args.representative_ids))
    visual_paths = make_visualizations(
        pairs, selected_ids, centers_lab, palette_rgb, cluster_shares, visual_dir
    )

    dataset_files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    documentation = [
        path.relative_to(dataset_root).as_posix()
        for path in dataset_files
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        or path.name.lower().startswith("readme")
    ]
    unmatched_x = sorted(x_images.keys() - y_images.keys())
    unmatched_y = sorted(y_images.keys() - x_images.keys())
    top_colors_summary = ", ".join(
        f"`{row['rgb']}` ({int(row['pixel_count']):,} px)"
        for row in global_color_rows[:12]
    )

    report_path = report_dir / "mask_analysis.md"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# Floor Plan Mask Analysis\n\n")
        report.write("## Dataset and pairing\n\n")
        report.write(f"- X images: {len(x_images)} PNG files with numeric IDs.\n")
        report.write(f"- Y masks: {len(y_images)} PNG files with numeric IDs.\n")
        report.write(f"- Matched numeric pairs: {len(shared_ids)}. Y suffix `(2)` was normalized.\n")
        report.write(f"- Unmatched X IDs: {unmatched_x or 'none'}.\n")
        report.write(f"- Unmatched Y IDs: {unmatched_y or 'none'}.\n")
        report.write(
            f"- Identical paired dimensions: {pair_dimension_matches}/{len(pairs)}.\n"
        )
        report.write(
            "- X dimensions (width x height, count): "
            + "; ".join(
                f"{width}x{height}: {count}"
                for (width, height), count in sorted(dimensions.items())
            )
            + ".\n"
        )
        report.write(
            f"- Filename extensions: `.png` in both folders. Encoded formats: "
            f"X {dict(x_formats)}, Y {dict(y_formats)}; all images decoded successfully.\n\n"
        )

        report.write("## Exact RGB frequencies\n\n")
        report.write(
            f"- Exact RGB colors observed across Y: {len(exact_global):,}; anti-aliasing creates "
            "many near-color variants.\n"
        )
        report.write(
            f"- Top exact RGB colors across all masks: {top_colors_summary}.\n"
        )
        report.write(
            f"- Per-mask top 10 exact colors for all {len(pairs)} masks are in "
            f"`{per_mask_csv.name}`; the global top 50 are in `{global_csv.name}`.\n\n"
        )

        report.write("## Grouped palette\n\n")
        report.write(
            f"OpenCV K-means grouped a deterministic, pixel-area-weighted sample of "
            f"{len(sample_rgb):,} pixels in CIE Lab space into {args.clusters} clusters "
            f"(seed {args.seed}). Representative RGB values and sample shares:\n\n"
        )
        report.write("| Rank | Representative RGB | Sample share |\n|---:|---|---:|\n")
        for row in cluster_rows:
            report.write(
                f"| {row['rank']} | `({row['representative_rgb']})` | {float(row['sample_percent']):.4f}% |\n"
            )
        report.write("\nThese are approximate color groups, not class IDs.\n\n")

        report.write("## Class mapping and evidence\n\n")
        report.write(
            "- Confirmed semantic class mappings: none. No inspected local README, metadata, "
            "documentation, or annotation/label files establish a color-to-class mapping.\n"
        )
        report.write(
            "- Supported observation: Y contains spatially aligned, multi-color RGB images that "
            "visually behave like region-segmentation targets; exact RGB variation at boundaries "
            "is consistent with anti-aliasing.\n"
        )
        report.write(
            "- Unidentified: which colors denote walls, doors, windows, rooms, or other components. "
            "Visual resemblance alone is not reliable mapping evidence.\n"
        )
        report.write(
            f"- Non-image files under the dataset root: {documentation or 'none found'}.\n\n"
        )

        report.write("## Recommended next step\n\n")
        report.write(
            "Obtain the dataset publisher's authoritative legend or annotation schema and "
            "verify it against several paired examples before defining any color-to-class "
            "mapping. Keep extraction work paused until that mapping is supported.\n\n"
        )

        report.write("## Visualizations\n\n")
        for path in visual_paths:
            report.write(f"- `../visualizations/{path.name}`\n")
        report.write("\n## Reproducibility\n\n")
        report.write(
            "Run `python src/mask_analyzer.py` from the project root. Options include "
            "`--representative-ids`, `--clusters`, `--sample-budget`, and `--seed`. "
            "Exact per-mask RGB counts are exhaustive; Lab cluster shares are estimated from "
            "a deterministic sample weighted by each mask's pixel area.\n"
        )

    print(f"Analyzed {len(pairs)} pairs; exact Y colors: {len(exact_global):,}")
    print(f"Report: {report_path}")
    print(f"CSV: {per_mask_csv}")
    print(f"CSV: {global_csv}")
    print(f"CSV: {cluster_csv}")
    for path in visual_paths:
        print(f"Visualization: {path}")


if __name__ == "__main__":
    main()