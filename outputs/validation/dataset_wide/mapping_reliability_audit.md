# Dataset-Wide Candidate Mapping Reliability Audit

Audited 121 matched X/Y pairs across the dataset. Unmatched X IDs: none; unmatched Y IDs: none. No source image or mask was modified. Visual sheets were generated for 40 representative/outlier pairs.

## Matching method

Each mask pixel is converted to OpenCV's 8-bit Lab space and compared with the six proposed RGB palette centers. The nearest candidate is accepted only when its Euclidean Lab distance is <= 12. Pixels whose nearest and second-nearest candidates differ by <= 3 are withheld as ambiguous. This is a conservative anti-aliasing tolerance, not a CIEDE2000 measurement or a semantic classifier. Exact-center coverage is reported separately. Outliers are identified per candidate using Tukey's 1.5 x IQR rule on Lab-matched per-image percentages.

## Dataset-Wide Coverage and Reliability

Across 1,594,099,645 pixels from 121 masks, 0 (0.0000%) were withheld as near-ties, 3,319,356 (0.2082%) were outside the palette tolerance, and 3,020,805 (0.1895%) exactly matched the unassigned red RGB.

| Candidate | RGB | Lab-matched pixels | Exact RGB pixels | Median per-image Lab share | Zero-coverage pairs | IQR outliers | Reliability assessment | Status |
|---|---|---:|---:|---:|---:|---|---|---|
| Room | `(255, 142, 70)` | 78,169,899 (4.904%) | 77,846,770 (4.883%) | 0.959% | 0 | 7 (31.87%), 29 (32.53%) | Low: room-colored coverage has strong cross-pair outliers. | **CANDIDATE** |
| Wall | `(175, 216, 248)` | 26,251,710 (1.647%) | 25,495,796 (1.599%) | 2.505% | 0 | 7 (5.81%), 9 (4.39%), 103 (0.43%), 104 (0.38%), 109 (0.45%), 110 (0.47%), 111 (0.45%), 112 (0.48%), 113 (0.30%), 114 (0.47%), 115 (0.35%), 116 (0.47%), 117 (0.31%), 120 (4.86%), 121 (4.75%) | Color coverage is measurable; semantic identity remains unverified. | **CANDIDATE** |
| Door | `(246, 189, 15)` | 8,254,460 (0.518%) | 8,130,188 (0.510%) | 0.543% | 0 | 40 (20.87%) | Color coverage is measurable; semantic identity remains unverified. | **CANDIDATE** |
| Window | `(139, 186, 0)` | 4,991,585 (0.313%) | 4,873,869 (0.306%) | 0.240% | 0 | 9 (1.24%), 26 (1.37%), 27 (0.92%), 33 (0.96%), 36 (1.44%), 38 (0.94%), 44 (1.25%), 45 (1.03%), 46 (1.16%), 61 (0.89%), 67 (14.72%), 70 (3.23%), 72 (3.13%), 87 (0.91%) | Color coverage is measurable; semantic identity remains unverified. | **CANDIDATE** |
| Separation | `(0, 142, 142)` | 92,691,728 (5.815%) | 92,312,538 (5.791%) | 0.930% | 24 | none | Very low: pair 61 has a large separation-colored filled region. | **CANDIDATE** |
| Unknown/background | `(0, 0, 0)` | 1,380,420,907 (86.596%) | 1,380,420,907 (86.596%) | 79.023% | 0 | 7 (60.20%) | Low semantic confidence: black is common but may include canvas/linework. | **CANDIDATE** |

The exact red color `(214, 70, 70)` occurs in 3,020,805 pixels and remains unmatched; no semantic meaning is assigned. Parking door remains **UNKNOWN**.

## Distribution Outliers

- Room: Q1=0.173%, Q3=12.802%, Tukey bounds [-18.771%, 31.746%]; outliers: 7 (31.873%), 29 (32.530%).
- Wall: Q1=2.000%, Q3=2.945%, Tukey bounds [0.583%, 4.362%]; outliers: 7 (5.810%), 9 (4.394%), 103 (0.431%), 104 (0.378%), 109 (0.446%), 110 (0.465%), 111 (0.450%), 112 (0.476%), 113 (0.300%), 114 (0.465%), 115 (0.346%), 116 (0.470%), 117 (0.313%), 120 (4.860%), 121 (4.753%).
- Door: Q1=0.217%, Q3=0.836%, Tukey bounds [-0.713%, 1.766%]; outliers: 40 (20.869%).
- Window: Q1=0.138%, Q3=0.433%, Tukey bounds [-0.305%, 0.876%]; outliers: 9 (1.235%), 26 (1.370%), 27 (0.916%), 33 (0.956%), 36 (1.440%), 38 (0.942%), 44 (1.246%), 45 (1.028%), 46 (1.164%), 61 (0.887%), 67 (14.715%), 70 (3.231%), 72 (3.134%), 87 (0.906%).
- Separation: Q1=0.064%, Q3=19.180%, Tukey bounds [-28.611%, 47.855%]; outliers: none.
- Unknown/background: Q1=75.044%, Q3=84.924%, Tukey bounds [60.223%, 99.744%]; outliers: 7 (60.204%).

## Room/Separation and Convention Checks

Pairs with orange Room >5% and teal Separation <1%: [6, 7, 8, 10, 15, 16, 19, 29, 31, 42, 47, 52, 56, 57, 58, 64, 65, 68, 70, 71, 72, 73, 75, 76, 80, 81, 84, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 99, 120, 121].

Pairs with teal Separation >5% and orange Room <1%: [1, 2, 3, 4, 5, 9, 11, 12, 13, 14, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 59, 60, 61, 62, 63, 66, 69, 77, 78, 79, 82, 83, 86, 93, 98, 118, 119].

This opposing distribution pattern is consistent with a possible image-level palette/convention change or an incorrect semantic assignment. It does not by itself prove that the two categories literally switch; room contents and lack of original SVGs remain confounders.

The largest non-background candidate color by pixel share is split across images as follows: Room: 59, Wall: 1, Door: 1, Window: 1, Separation: 59. Room and Separation are tied at 59 images each, while Door, Window, and Wall lead in pairs 40, 67, and 85 respectively. These winner labels are only candidate palette names. The broad interior fills in pair 40 (Door yellow, 20.869%) and pair 67 (Window green, 14.715%) are visually suspicious semantic false positives.

The candidate palette colors recur across the dataset, but class shares and unmatched-color rates vary by image. The opposing Room/Separation and other dominant-color regimes are consistent with multiple annotation styles or inconsistent per-image color conventions, but cannot distinguish those from different plan content or false candidate mappings without original SVG annotations. They do not prove a literal Room/Separation role switch. No pixel is assigned to more than one candidate; near-ties are withheld.

## Other Suspicious Cases and Limitations

Exact-color and Lab-matched percentages are separate: the latter includes near anti-aliased shades within the configured tolerance. These are color-coverage statistics, not semantic validation. The third-party notebook hard-codes its display palette, and the original SVG archive was unavailable, so no class mapping is confirmed.

## Visual sheets

- [Pair 1](visuals/pair_001_validation.png)
- [Pair 7](visuals/pair_007_validation.png)
- [Pair 8](visuals/pair_008_validation.png)
- [Pair 9](visuals/pair_009_validation.png)
- [Pair 16](visuals/pair_016_validation.png)
- [Pair 24](visuals/pair_024_validation.png)
- [Pair 26](visuals/pair_026_validation.png)
- [Pair 27](visuals/pair_027_validation.png)
- [Pair 29](visuals/pair_029_validation.png)
- [Pair 31](visuals/pair_031_validation.png)
- [Pair 33](visuals/pair_033_validation.png)
- [Pair 36](visuals/pair_036_validation.png)
- [Pair 38](visuals/pair_038_validation.png)
- [Pair 40](visuals/pair_040_validation.png)
- [Pair 44](visuals/pair_044_validation.png)
- [Pair 45](visuals/pair_045_validation.png)
- [Pair 46](visuals/pair_046_validation.png)
- [Pair 48](visuals/pair_048_validation.png)
- [Pair 56](visuals/pair_056_validation.png)
- [Pair 61](visuals/pair_061_validation.png)
- [Pair 67](visuals/pair_067_validation.png)
- [Pair 70](visuals/pair_070_validation.png)
- [Pair 72](visuals/pair_072_validation.png)
- [Pair 83](visuals/pair_083_validation.png)
- [Pair 87](visuals/pair_087_validation.png)
- [Pair 91](visuals/pair_091_validation.png)
- [Pair 101](visuals/pair_101_validation.png)
- [Pair 103](visuals/pair_103_validation.png)
- [Pair 104](visuals/pair_104_validation.png)
- [Pair 109](visuals/pair_109_validation.png)
- [Pair 110](visuals/pair_110_validation.png)
- [Pair 111](visuals/pair_111_validation.png)
- [Pair 112](visuals/pair_112_validation.png)
- [Pair 113](visuals/pair_113_validation.png)
- [Pair 114](visuals/pair_114_validation.png)
- [Pair 115](visuals/pair_115_validation.png)
- [Pair 116](visuals/pair_116_validation.png)
- [Pair 117](visuals/pair_117_validation.png)
- [Pair 120](visuals/pair_120_validation.png)
- [Pair 121](visuals/pair_121_validation.png)

Each sheet includes the source plan, source mask, candidate-colored overlay, and six individual candidate masks. Magenta marks ambiguous pixels; gray marks pixels outside the candidate palette tolerance. Full per-pair exact and Lab coverage is in `pair_mapping_coverage.csv`.

## Recommendation

Do not use the unverified semantic palette as the sole basis for extraction. A hybrid approach (C) is the safest next research direction: preserve mask color evidence with uncertainty reporting, then cross-check geometry with computer vision and the original SVG schema when obtainable. No extraction was started.
