# STARS human evaluation

This folder contains individual human ratings of baseline and STARS scripts generated with LLaVA-Video-7B-Qwen2, InternVL2.5-8B, and VideoLLaMA2-7B-16F. The evaluation covers the same 60 videos for all three models: 20 from LongVideoBench, 20 from CG-Bench, and 20 from Video-MME.

Each CSV contains 120 records, one for each video and method combination, with five scores per record. The complete collection contains 4,320 records and 21,600 individual scores.

## Files

```text
human_evaluation/
├── README.md
├── ratings/
│   ├── 1.csv
│   ├── ...
│   └── 36.csv
└── original_xlsx.zip
```

- `ratings/1.csv` through `ratings/36.csv`: UTF-8, comma-separated exports with a single header row.
- `original_xlsx.zip`: the corresponding `1.xlsx` through `36.xlsx`, preserved without changes. CSV and XLSX files with the same number contain the same scores.

## Evaluation protocol

The study involved 36 volunteers, divided into three groups of 12. Each group evaluated scripts generated with one model. After watching each video, evaluators scored the baseline and STARS scripts on five dimensions: visual grounding, temporal consistency, completeness, readability, and overall preference. Method labels were concealed, and script presentation order was randomized for blind assessment. Each evaluator's scores are provided in a separate CSV file.

## CSV fields

Each row contains one evaluator's scores for one video, one model, and one method. The combination of `rater_id`, `model`, `dataset`, `video_id`, and `method` uniquely identifies a row.

| Field | Description |
| --- | --- |
| `rater_id` | Anonymous evaluator identifier: `R01` for `1.csv`, through `R36` for `36.csv`. |
| `model` | Model used to generate the evaluated scripts. |
| `dataset` | `LongVideoBench`, `CG-Bench`, or `Video-MME`. |
| `video_id` | Video identifier copied from the source column `Original ID`. Read this field as text and use it together with `dataset`. |
| `method` | `Baseline` or `STARS`, as recorded in the source sheet. |
| `visual_grounding` | Score for correspondence between the script and visual evidence. |
| `temporal_consistency` | Score for consistency with the sequence of events. |
| `completeness` | Score for coverage of relevant video content. |
| `readability` | Score for clarity and ease of reading. |
| `overall_preference` | Overall preference score for the script. |

The five score fields contain integers from 1 to 5. The rating scale is 1 = poor, 2 = average, 3 = good, 4 = very good, and 5 = excellent. Higher scores indicate better ratings for all five dimensions.

## File numbering and models

There are 12 evaluators per model. The table below maps every CSV file to its model. Each file uses its own `rater_id`; entries on the same table row correspond to different evaluators across models.

| Index within model group | LLaVA-Video-7B-Qwen2 | InternVL2.5-8B | VideoLLaMA2-7B-16F |
| --- | --- | --- | --- |
| 1 | [1.csv](ratings/1.csv) | [2.csv](ratings/2.csv) | [3.csv](ratings/3.csv) |
| 2 | [4.csv](ratings/4.csv) | [5.csv](ratings/5.csv) | [6.csv](ratings/6.csv) |
| 3 | [7.csv](ratings/7.csv) | [8.csv](ratings/8.csv) | [9.csv](ratings/9.csv) |
| 4 | [10.csv](ratings/10.csv) | [11.csv](ratings/11.csv) | [12.csv](ratings/12.csv) |
| 5 | [13.csv](ratings/13.csv) | [14.csv](ratings/14.csv) | [15.csv](ratings/15.csv) |
| 6 | [16.csv](ratings/16.csv) | [17.csv](ratings/17.csv) | [18.csv](ratings/18.csv) |
| 7 | [19.csv](ratings/19.csv) | [20.csv](ratings/20.csv) | [21.csv](ratings/21.csv) |
| 8 | [22.csv](ratings/22.csv) | [23.csv](ratings/23.csv) | [24.csv](ratings/24.csv) |
| 9 | [25.csv](ratings/25.csv) | [26.csv](ratings/26.csv) | [27.csv](ratings/27.csv) |
| 10 | [28.csv](ratings/28.csv) | [29.csv](ratings/29.csv) | [30.csv](ratings/30.csv) |
| 11 | [31.csv](ratings/31.csv) | [32.csv](ratings/32.csv) | [33.csv](ratings/33.csv) |
| 12 | [34.csv](ratings/34.csv) | [35.csv](ratings/35.csv) | [36.csv](ratings/36.csv) |

## Conversion and aggregation

Dataset names and video identifiers that occupied merged cells in Excel are repeated on every corresponding CSV row. Source column names are standardized to the field names above, and model and anonymous evaluator identifiers are added explicitly. Formatting-only template annotations are omitted from the CSV exports. The original row order and all score values are retained. No scores are altered, rounded, averaged, or omitted during conversion.

To calculate a mean score for each script and dimension, group the records by `model`, `dataset`, `video_id`, and `method`, then average the relevant score across the 12 sheets for that model. Baseline and STARS records can be paired using `model`, `dataset`, and `video_id` after this averaging step.

This folder contains score data and video identifiers, not the video files or generated scripts.
