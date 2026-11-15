# 20-Question Evaluation

## Summary

- Total questions: 20
- Correct database-query answers: 15
- Incorrect database-query answers: 0
- Partially correct answers: 0
- Correct unsupported-question refusals: 5
- Accuracy on evaluable database queries: 100.0%
- Live Gemini answers evaluated: 0 (no API key configured).

Expected values are computed from the actual imported SQLite records. The five refusal tests exercise the deterministic data-capability guard, not a live Gemini response. Candidate counts and areas are not ground-truth accuracy measurements.

## Per-question results

| ID | Group | Question | Operation | Status |
|---|---|---|---|---|
| E01 | cross-plan count | How many floor plans are in the database? | list_floor_plans | passed |
| E02 | count | How many candidate rooms are detected in plan 8? | count_elements | passed |
| E03 | count | How many candidate doors are in plan 31? | count_elements | passed |
| E04 | count | How many candidate windows are in plan 61? | count_elements | passed |
| E05 | count | How many candidate walls are in plan 8? | count_elements | passed |
| E06 | area | What is the largest candidate room region in plan 8? | largest_room | passed |
| E07 | area | What is the largest candidate room region in plan 31? | largest_room | passed |
| E08 | area | What is the total candidate room area in pixels for plan 61? | sum_room_pixel_area | passed |
| E09 | area | What is the total candidate room area in pixels across all plans? | sum_room_pixel_area | passed |
| E10 | comparison | Compare candidate room counts in plans 8 and 31. | compare_room_counts | passed |
| E11 | cross-plan comparison | Which plans have the most candidate room regions? | most_rooms | passed |
| E12 | location | Show the bounding box of a candidate room in plan 8. | get_element | passed |
| E13 | summary | Summarize candidate detections in plan 61. | get_floor_plan_summary | passed |
| E14 | geometry | Which candidate room boxes are adjacent in plan 31? | find_adjacent_rooms | passed |
| E15 | count | How many candidate elements of each category are in plan 61? | get_floor_plan_summary | passed |
| E16 | unsupported | What is the total room area in square metres for plan 8? | refusal guard | correct_refusal |
| E17 | unsupported | What are the room names and which one is the kitchen? | refusal guard | correct_refusal |
| E18 | unsupported | How many parking doors are in plan 31? | refusal guard | correct_refusal |
| E19 | unsupported | What material are the walls made from? | refusal guard | correct_refusal |
| E20 | unsupported | What architectural class does the red mask color represent? | refusal guard | correct_refusal |

## Limitations

The evaluation verifies database query correctness and deterministic refusals only. It does not evaluate natural-language accuracy of Gemini because GEMINI_API_KEY is not configured. No expected semantic labels or physical areas were invented; the evaluated data are candidate detections and pixel measurements.
