# Candidate Color-Mapping Validation

Validated local pairs 8, 31, 61, 91, 121 (5 samples). No source image or mask was modified. Pair 8 is included because it is the one strong image match to the CVC-FP mirror example; pairs 31, 61, 91, and 121 provide additional visual variation.

## Matching method

Each mask pixel is converted to OpenCV's 8-bit Lab space and compared with the six proposed RGB palette centers. The nearest candidate is accepted only when its Euclidean Lab distance is <= 12. Pixels whose nearest and second-nearest candidates differ by <= 3 are withheld as ambiguous. This is a conservative anti-aliasing tolerance, not a CIEDE2000 measurement or a semantic classifier. Exact-center coverage is reported separately.

## Per-pair candidate coverage

| Pair | Room | Wall | Door | Window | Separation | Unknown/background | Ambiguous | Unmatched | Red (unknown) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 14.705% | 4.067% | 0.158% | 0.232% | 0.483% | 80.334% | 0.000% | 0.022% | 0.000% |
| 31 | 18.587% | 3.402% | 0.857% | 0.574% | 0.144% | 76.400% | 0.000% | 0.036% | 0.000% |
| 61 | 0.135% | 2.435% | 0.293% | 0.887% | 17.241% | 78.736% | 0.000% | 0.273% | 0.242% |
| 91 | 22.161% | 3.840% | 1.157% | 0.426% | 0.180% | 72.201% | 0.000% | 0.035% | 0.000% |
| 121 | 28.921% | 4.753% | 0.522% | 0.185% | 0.000% | 65.566% | 0.000% | 0.052% | 0.000% |

## Candidate evidence and confidence

| Candidate | RGB | Matched pixels | Exact RGB pixels | Visual validation evidence | Confidence | Status |
|---|---|---:|---:|---|---|---|
| Room | `(255, 142, 70)` | 6,312,073 (17.200%) | 6,288,880 (17.137%) | CVC-FP mirror notebook palette plus visual alignment in pair 8; original SVG unavailable. | Medium for one matched sample; low dataset-wide | **CANDIDATE** |
| Wall | `(175, 216, 248)` | 1,330,649 (3.626%) | 1,293,850 (3.526%) | CVC-FP mirror notebook palette and outline-like regions in pair 8; no original SVG comparison. | Low to medium for one matched sample | **CANDIDATE** |
| Door | `(246, 189, 15)` | 248,083 (0.676%) | 244,541 (0.666%) | CVC-FP mirror notebook palette and small opening-aligned marks in pair 8; no SVG comparison. | Low to medium for one matched sample | **CANDIDATE** |
| Window | `(139, 186, 0)` | 184,806 (0.504%) | 179,670 (0.490%) | CVC-FP mirror notebook palette and small boundary-aligned marks in pair 8; no SVG comparison. | Low to medium for one matched sample | **CANDIDATE** |
| Separation | `(0, 142, 142)` | 1,538,098 (4.191%) | 1,532,970 (4.177%) | CVC-FP mirror notebook palette and a small separating region in pair 8; no SVG comparison. | Low for one matched sample | **CANDIDATE** |
| Unknown/background | `(0, 0, 0)` | 27,049,195 (73.708%) | 27,049,195 (73.708%) | Mirror notebook labels black as Unknown; it dominates the canvas in local masks. | Low | **CANDIDATE** |

The exact red color `(214, 70, 70)` occurs in the selected samples at 21,018 pixels (0.057%). It is deliberately left unmatched and has no assigned semantic meaning.

## Suspicious cases and limitations

- Ambiguous near-ties withheld: 0 pixels (0.000% of selected pixels). Nearest-candidate masks are mutually exclusive by construction; any near-tie is excluded rather than counted in two classes.
- Pixels outside the candidate distance limit: 34,837 (0.095%). These include the red group and other colors not represented by the candidate palette.
- **Potential mapping false positive, pair 61:** Room orange is 0.135% while Separation teal is 17.241%. The teal mask appears as broad filled regions in the visual sheet, unlike the small separator-like region in pair 8. This conflicts with a stable global Room/Separation mapping and may indicate per-image recoloring or an incorrect candidate assignment.
- Black is a candidate Unknown/background label and occupies most of the masks; it may include unlabeled canvas or retained drawing content, so its large coverage is not evidence of an architectural category.
- Coverage measures color proximity only. A correctly matched color can still be a semantic false positive if the third-party notebook's manual color assignment does not reflect the original SVG class. The original SVG archive was unavailable, so none of these candidates is confirmed. Parking door remains **UNKNOWN**.

## Visual sheets

- [Pair 8](pair_008_validation.png)
- [Pair 31](pair_031_validation.png)
- [Pair 61](pair_061_validation.png)
- [Pair 91](pair_091_validation.png)
- [Pair 121](pair_121_validation.png)

Each sheet includes the source plan, source mask, candidate-colored overlay, and six individual candidate masks. Magenta marks ambiguous pixels; gray marks pixels outside the candidate palette tolerance.
