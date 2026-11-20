# Data Directory

This directory holds feature CSVs produced by the extraction pipeline.

## Sub-directories

| Path | Contents |
|---|---|
| `features/` | Extracted feature CSVs (`phone_features.csv`, `wrist_features.csv`) |

## Raw Dataset

The raw UMAFall CSV files live **outside** this repository at the path set by
`DATASET_DIR` in your `.env` file (default: `d:/Farhan/Assignment/Dataset/`).

## File Naming Convention (UMAFall)

```
UMAFall_Subject_<NN>_<ADL|Fall>_<Description>_<Trial>_<Date>.csv
```

### File Format
- Lines starting with `%` are metadata / the data header.
- The data header line contains `TimeStamp`.
- Data rows are semicolon-delimited: `timestamp;sample_no;x;y;z;sensor_type;sensor_id`

### Sensor Types
| Code | Sensor |
|---|---|
| 0 | Accelerometer |
| 1 | Gyroscope |
| 2 | Magnetometer |

### Sensor IDs / Positions
| ID | Position |
|---|---|
| 0 | RIGHTPOCKET (phone) |
| 1 | CHEST |
| 2 | WAIST |
| 3 | WRIST |
| 4 | ANKLE |

### Label
`% Type of Movement: TRUE` → **Fall**  
`% Type of Movement: FALSE` → **ADL** (Activities of Daily Living)
