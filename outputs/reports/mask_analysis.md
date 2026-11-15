# Floor Plan Mask Analysis

## Dataset and pairing

- X images: 121 PNG files with numeric IDs.
- Y masks: 121 PNG files with numeric IDs.
- Matched numeric pairs: 121. Y suffix `(2)` was normalized.
- Unmatched X IDs: none.
- Unmatched Y IDs: none.
- Identical paired dimensions: 121/121.
- X dimensions (width x height, count): 1027x1369: 1; 1298x1506: 1; 1318x1229: 1; 1479x1486: 1; 1497x1462: 1; 1621x1349: 1; 1943x1987: 1; 1950x1867: 1; 2218x2227: 1; 2479x3508: 45; 2480x3508: 22; 2550x3300: 4; 3508x2479: 7; 3508x2480: 16; 5671x7383: 18.
- Filename extensions: `.png` in both folders. Encoded formats: X {'PNG': 117, 'JPEG': 4}, Y {'PNG': 121}; all images decoded successfully.

## Exact RGB frequencies

- Exact RGB colors observed across Y: 15,697; anti-aliasing creates many near-color variants.
- Top exact RGB colors across all masks: `0,0,0` (1,380,420,907 px), `0,142,142` (92,312,538 px), `255,142,70` (77,846,770 px), `175,216,248` (25,495,796 px), `246,189,15` (8,130,188 px), `139,186,0` (4,873,869 px), `214,70,70` (3,020,805 px), `0,143,143` (181,086 px), `0,141,141` (90,557 px), `175,215,248` (75,746 px), `174,216,248` (63,229 px), `255,143,70` (53,641 px).
- Per-mask top 10 exact colors for all 121 masks are in `mask_top_rgb_colors.csv`; the global top 50 are in `dataset_top_rgb_colors.csv`.

## Grouped palette

OpenCV K-means grouped a deterministic, pixel-area-weighted sample of 1,000,003 pixels in CIE Lab space into 8 clusters (seed 2026). Representative RGB values and sample shares:

| Rank | Representative RGB | Sample share |
|---:|---|---:|
| 1 | `(0,0,0)` | 86.6233% |
| 2 | `(0,142,142)` | 5.8146% |
| 3 | `(255,142,70)` | 4.8965% |
| 4 | `(175,216,248)` | 1.6462% |
| 5 | `(246,189,15)` | 0.5003% |
| 6 | `(139,186,0)` | 0.3285% |
| 7 | `(214,70,70)` | 0.1862% |
| 8 | `(201,180,145)` | 0.0044% |

These are approximate color groups, not class IDs.

## Class mapping and evidence

- Confirmed semantic class mappings: none. No inspected local README, metadata, documentation, or annotation/label files establish a color-to-class mapping.
- Supported observation: Y contains spatially aligned, multi-color RGB images that visually behave like region-segmentation targets; exact RGB variation at boundaries is consistent with anti-aliasing.
- Unidentified: which colors denote walls, doors, windows, rooms, or other components. Visual resemblance alone is not reliable mapping evidence.
- Non-image files under the dataset root: none found.

## Recommended next step

Obtain the dataset publisher's authoritative legend or annotation schema and verify it against several paired examples before defining any color-to-class mapping. Keep extraction work paused until that mapping is supported.

## Visualizations

- `../visualizations/pair_001_comparison.png`
- `../visualizations/pair_031_comparison.png`
- `../visualizations/pair_061_comparison.png`
- `../visualizations/pair_091_comparison.png`
- `../visualizations/pair_121_comparison.png`
- `../visualizations/dataset_dominant_palette.png`

## Reproducibility

Run `python src/mask_analyzer.py` from the project root. Options include `--representative-ids`, `--clusters`, `--sample-budget`, and `--seed`. Exact per-mask RGB counts are exhaustive; Lab cluster shares are estimated from a deterministic sample weighted by each mask's pixel area.
