# Manitoba Wildfire Risk Intelligence — Technical Project Documentation

**Document purpose:** Living technical record of the project methodology, processing decisions, validation results, modelling design, and major outputs.  
**Current checkpoint:** Data pipeline, feature engineering, chronological split, and historical baselines complete.  
**Study period:** 2005–2025  
**Study area:** Manitoba, Canada  
**Spatial unit:** 10 km × 10 km grid cell  
**Forecast horizon:** One day ahead

---

## 1. Why This Document Exists

The public `README.md` provides a concise overview of the project. This file preserves the deeper technical history: what was done, why particular design decisions were made, what was validated, what problems were encountered, and what outputs were produced.

It should be updated at major checkpoints so that modelling decisions can be traced later without relying on notebook output, terminal history, or memory.

---

## 2. Current Research Design

The project estimates the probability that at least one new wildfire will be recorded in a Manitoba 10-km grid cell on the following day.

The modelling relationship is:

```text
Predictor information available through day t
                    ↓
Wildfire occurrence in the same grid cell on day t + 1
```

The primary binary target is:

```text
FIRE_OCCURRED_NEXT_DAY = 1
    At least one fire is recorded in the grid cell on FORECAST_DATE

FIRE_OCCURRED_NEXT_DAY = 0
    No fire is recorded in the grid cell on FORECAST_DATE
```

A secondary count field, `FIRE_COUNT_NEXT_DAY`, preserves the number of individual fires represented by a positive grid-cell day.

### Current fixed scope

| Design element | Decision |
|---|---|
| Geographic area | Manitoba |
| Historical period | 2005–2025 |
| Spatial resolution | 10 km × 10 km |
| Grid cells | 6,501 |
| Forecast season | April 1–October 31 |
| Weather-support season | March 1–October 31 |
| Forecast horizon | 1 day |
| Main task | Binary classification |
| Train | 2005–2019 |
| Validation | 2020–2022 |
| Test | 2023–2025 |

---

## 3. Source Data

### 3.1 Canadian National Fire Database

The historical fire source contains Manitoba-reported wildfire point records with dates, coordinates, fire sizes, causes, and related attributes.

Initial Manitoba subset:

| Measurement | Result |
|---|---:|
| Manitoba records, 1959–2025 | 28,797 |
| Exact usable dates | 25,940 |
| Missing exact dates | 2,857 |
| Records in 2005–2025 before exact duplicate removal | 7,933 |
| Exact duplicate rows removed | 2 |
| Cleaned 2005–2025 records | 7,931 |
| Zero-size records retained | 153 |

The daily modelling period begins in 2005 because missing exact dates are concentrated in older portions of the archive, particularly around 2000–2003.

### 3.2 Manitoba Provincial Boundary

The official provincial boundary is used to validate coordinates and define the modelling geometry.

Of the 7,931 cleaned 2005–2025 records:

- 7,915 fall within or are covered by the Manitoba boundary and are eligible for grid modelling.
- 16 Manitoba-reported records fall slightly outside the official modelling boundary, by no more than roughly 9 km.
- Those 16 records are retained in the cleaned historical dataset for transparency but excluded from the grid-based modelling target.

### 3.3 ERA5-Land

Hourly ERA5-Land variables currently used:

- 2-metre temperature (`t2m`)
- 2-metre dew-point temperature (`d2m`)
- 10-metre eastward wind (`u10`)
- 10-metre northward wind (`v10`)
- Total precipitation (`tp`)

Derived daily variables:

- `TEMP_MAX_C`
- `TEMP_MIN_C`
- `TEMP_MEAN_C`
- `RH_MIN_PCT`
- `RH_MEAN_PCT`
- `WIND_MAX_MS`
- `WIND_MEAN_MS`
- `PRECIPITATION_MM`

Temperature and dew point are converted from Kelvin to Celsius, wind speed is calculated from the U/V vector components, precipitation is converted from metres to millimetres, and relative humidity is derived from air temperature and dew point.

---

## 4. Wildfire Data Preparation

Primary exploratory notebook:

```text
notebooks/01_fire_data_exploration.ipynb
```

Major QA/QC steps included:

- Filtering source records to Manitoba reporting agency records.
- Parsing and auditing dates.
- Reviewing missing exact dates.
- Inspecting latitude and longitude ranges.
- Auditing fire sizes.
- Retaining zero-size fires rather than automatically removing them.
- Reviewing fire causes.
- Removing two exact duplicate rows.
- Validating fire locations against the provincial boundary.

### Cause summary after duplicate removal

| Cause | Cleaned records |
|---|---:|
| Natural | 4,659 |
| Human | 3,255 |
| Unknown | 17 |
| Total | 7,931 |

Cause is retained for exploratory analysis. It is not used as an input feature for general next-day fire prediction because cause is normally assigned after an ignition is observed.

---

## 5. Manitoba 10-km Grid

Primary notebook:

```text
notebooks/02_manitoba_grid.ipynb
```

The grid is generated in a metre-based projected coordinate reference system (`EPSG:3347`) using 10,000-metre cells, then clipped/retained based on intersection with Manitoba.

Validated grid results:

| Measurement | Result |
|---|---:|
| Grid cells intersecting Manitoba | 6,501 |
| Eligible fires assigned to a grid cell | 7,915 |
| Unmatched eligible fires | 0 |
| Cells containing ≥1 historical fire | 2,593 |
| Maximum fires in one cell | 140 |

Important generated files include:

```text
data/interim/manitoba_grid_10km.parquet
data/interim/manitoba_fires_with_grid_2005_2025.parquet
```

---

## 6. Daily Fire Targets

Primary notebook:

```text
notebooks/03_daily_fire_targets.ipynb
```

Fires are aggregated by exact date and `GRID_ID`.

The positive target table contains only date-grid combinations with one or more recorded fires. Negative grid-days are created later when the positive target table is joined to the complete daily weather grid.

Validated complete target results:

| Measurement | Result |
|---|---:|
| Positive date-grid combinations | 7,356 |
| Individual fires represented | 7,915 |
| Date-grid combinations with multiple fires | 434 |
| Maximum fires in one grid cell on one day | 10 |
| Duplicate date-grid combinations | 0 |

Generated files:

```text
data/interim/manitoba_daily_fire_targets_2005_2025.parquet
data/interim/manitoba_grid_reference_10km.parquet
```

---

## 7. ERA5-Land Pilot Development

### 7.1 Initial pilot

Primary notebook:

```text
notebooks/04_era5_weather_pilot.ipynb
```

The pilot established:

- CDS API access.
- ERA5-Land hourly download structure.
- Temperature, relative humidity, wind, and precipitation conversion.
- UTC-to-Manitoba-local timestamp conversion.
- Daily aggregation.
- Spatial matching between Manitoba grid cells and ERA5-Land locations.

The initial two-day pilot produced:

| Measurement | Result |
|---|---:|
| Manitoba grid cells | 6,501 |
| Pilot dates | 2 |
| Grid-cell days | 13,002 |
| Missing weather after correction | 0 |
| Unique ERA5-Land locations used | 6,006 |
| Maximum nearest-valid-land distance | 35.03 km |

### 7.2 Water/coastal matching issue

Forty-four Manitoba grid cells initially lacked some instantaneous ERA5-Land variables because their nearest ERA5 coordinate corresponded to water or another invalid land point.

Solution:

1. Identify valid ERA5-Land locations.
2. Use geographic nearest-neighbor matching with a BallTree / haversine distance.
3. Match each Manitoba grid centroid to the nearest valid ERA5-Land point.
4. Retain match distance and quality information.

Generated matching table:

```text
data/interim/manitoba_grid_to_era5_land_matches.parquet
```

This produced complete weather coverage for all 6,501 Manitoba cells.

---

## 8. July 2024 and Full-Season Prototypes

### July 2024

Primary notebook:

```text
notebooks/05_era5_weather_july_2024.ipynb
```

Validated results:

| Measurement | Result |
|---|---:|
| Hourly timestamps including buffer | 768 |
| Complete July local dates | 31 |
| Grid-day rows | 201,531 |
| Missing weather | 0 |
| Positive fire grid-days | 111 |
| Individual fires | 118 |

### 2024 fire season

Primary notebook:

```text
notebooks/06_era5_weather_2024_fire_season.ipynb
```

This expanded the workflow to the full April–October fire season and added March as antecedent-weather support.

The final annual production design uses:

```text
March 1–October 31  → weather support
April 1–October 31  → forecast / target season
```

The March support window is necessary so April 1 predictions can have complete 30-day antecedent weather features.

---

## 9. Multi-Year ERA5-Land Automation

### 9.1 Download script

```text
src/data/download_era5_fire_season.py
```

Behavior:

- Processes one year at a time.
- Downloads complete monthly ERA5-Land files.
- Includes March through October plus the required November 1 buffer for local-time completeness during weather aggregation.
- Uses temporary `.part` files.
- Validates file size, variables, and expected timestamps.
- Retries recoverable CDS failures.
- Skips files that already exist and pass validation.

### 9.2 Processing script

```text
src/data/process_era5_fire_season.py
```

Behavior:

- Reads the annual hourly ERA5-Land files.
- Converts timestamps to `America/Winnipeg` local time.
- Aggregates hourly data into daily weather variables.
- Uses the validated grid-to-ERA5 matching table.
- Produces March–October weather-support tables.
- Produces April–October same-day weather/fire model tables.
- Writes Parquet outputs safely and validates them after writing.

### 9.3 Daylight-saving-time correction

The initial annual validator assumed every local day had exactly 24 hourly observations. This failed on daylight-saving transition dates.

The validator was replaced with a timezone-aware expected-hour calculation:

- Spring DST transition: 23-hour local day is valid.
- Fall DST transition: 25-hour local day is valid.
- Ordinary dates: 24 hours.

Examples validated during processing included:

- 2024-03-10: 23 hours
- 2005-04-03: 23 hours
- 2005-10-30: 25 hours

This was a pipeline validation issue, not missing source weather.

### 9.4 Batch runner

```text
scripts/run_era5_batch.sh
```

The batch runner processes a requested year range sequentially and is resumable because annual downloader/processor scripts validate and skip completed work.

Large batches were completed in groups before the final 2023–2025 run. Temporary CDS failures were retried successfully.

---

## 10. Multi-Year Weather Audit

Audit script:

```text
src/data/audit_multiyear_outputs.py
```

Audit file:

```text
outputs/tables/multiyear_data_audit_2005_2025.csv
```

Every annual weather and model table was independently checked for:

- File existence.
- Expected row count.
- Expected date range.
- Expected number of dates.
- 6,501 unique grid cells.
- Duplicate date-grid combinations.
- Missing weather values.
- Positive target counts.
- Individual fire totals.
- Agreement with the original positive-fire target table.

Final result:

```text
Years expected: 21
Years passed:   21
Years failed:   0
```

For each year, the March–October weather-support table contains:

```text
245 days × 6,501 cells = 1,592,745 rows
```

The April–October model table contains:

```text
214 days × 6,501 cells = 1,391,214 rows
```

Across the April–October season for all 21 years, 7,882 individual fires are represented.

---

## 11. Next-Day Feature Engineering

Feature script:

```text
src/features/build_weather_features.py
```

Batch runner:

```text
scripts/run_feature_batch.sh
```

Feature audit:

```text
src/features/audit_feature_outputs.py
```

### 11.1 Leakage-prevention design

The original annual model tables contain same-day weather and same-day fire targets. Those tables are useful intermediate products but are not passed directly into model training.

The feature-engineering script creates two explicit dates:

```text
PREDICTOR_DATE = day t
FORECAST_DATE  = day t + 1
```

For example:

```text
March 31 weather  → April 1 fire target
April 1 weather   → April 2 fire target
...
October 30 weather → October 31 fire target
```

Only information available through the predictor date is used to construct weather features.

### 11.2 Current feature set

Daily weather conditions on day t:

- `TEMP_MAX_C`
- `TEMP_MIN_C`
- `TEMP_MEAN_C`
- `RH_MIN_PCT`
- `RH_MEAN_PCT`
- `WIND_MAX_MS`
- `WIND_MEAN_MS`
- `PRECIPITATION_MM`

Antecedent precipitation:

- `PRECIP_3D_MM`
- `PRECIP_7D_MM`
- `PRECIP_30D_MM`

Recent temperature:

- `TEMP_MAX_3D_MEAN_C`
- `TEMP_MAX_7D_MEAN_C`

Recent humidity:

- `RH_MIN_3D_MEAN_PCT`
- `RH_MIN_7D_MEAN_PCT`

Recent wind:

- `WIND_MAX_3D_MEAN_MS`
- `WIND_MAX_7D_MEAN_MS`

Dry-spell feature:

- `DAYS_SINCE_RAIN`

A day is treated as a rain day when daily precipitation is at least 1 mm. The feature counts consecutive days below this threshold through predictor day t.

Seasonality:

- `DAY_OF_YEAR`
- `DOY_SIN`
- `DOY_COS`

`DAY_OF_YEAR` is based on the forecast date.

### 11.3 Feature validation

The 2024 pilot passed with:

```text
Rows:                              1,391,214
Forecast dates:                    214
Grid cells:                        6,501
Duplicate date-grid rows:          0
Missing feature values:            0
Positive next-day grid-cell days:  298
Individual fires:                  312
```

The batch was then run for all years.

Final multi-year feature audit:

```text
Years expected:                    21
Years passed:                      21
Years failed:                      0
Total feature rows:                29,215,494
Positive next-day grid-cell days:  7,324
Individual fires represented:      7,882
```

The difference between 7,356 positive grid-cell days in the full daily target table and 7,324 in the feature dataset is expected: 32 positive grid-cell days, representing 33 individual fires, occur outside the April–October modelling season.

---

## 12. Class Imbalance

Final feature dataset:

```text
29,215,494 total grid-day observations
7,324 positive observations
```

Overall positive prevalence is approximately:

```text
0.0251%
```

This means ordinary accuracy is not a meaningful primary metric. A model predicting no fire almost everywhere would achieve extremely high accuracy while failing to identify meaningful risk.

Primary evaluation therefore emphasizes:

- Average Precision / PR-AUC.
- Fire capture among the highest-risk observations.
- Precision.
- Recall.
- Calibration.
- Brier score.
- Threshold-specific false-alarm burden.
- ROC-AUC as a secondary ranking metric.

---

## 13. Chronological Model Split

Split script:

```text
src/models/create_split_manifest.py
```

Generated outputs:

```text
outputs/tables/model_split_manifest_2005_2025.csv
outputs/tables/model_split_summary_2005_2025.csv
```

The project does **not** use a random row-level train/test split.

Reason: observations from nearby grid cells and neighboring dates share weather and spatial context. Random splitting could place strongly related observations in both training and evaluation data, producing an unrealistically optimistic estimate of future-season performance.

Current split:

| Split | Years | Rows | Positive targets | Individual fires | Positive prevalence |
|---|---:|---:|---:|---:|---:|
| Train | 2005–2019 | 20,868,210 | 5,500 | 5,964 | 0.026356% |
| Validation | 2020–2022 | 4,173,642 | 822 | 863 | 0.019695% |
| Test | 2023–2025 | 4,173,642 | 1,002 | 1,055 | 0.024008% |

### Test-set rule

The 2023–2025 test period is considered sealed.

It may be counted for dataset bookkeeping, but it must **not** be used to:

- Choose features.
- Select a model family.
- Tune hyperparameters.
- Select probability thresholds.
- Decide whether a modelling change is an improvement.

Final test evaluation should occur only after model selection using training and validation periods.

---

## 14. Baseline 1 — Historical Spatial Risk

Script:

```text
src/models/evaluate_spatial_baseline.py
```

Training period:

```text
2005–2019
```

Validation period:

```text
2020–2022
```

Test period:

```text
NOT USED
```

### Method

For each `GRID_ID`, calculate its historical positive-fire frequency during training:

```text
historical positive grid-days / training observations for that grid cell
```

The resulting score knows **where** fires historically occur but has no weather or within-season timing information.

Training results:

| Measurement | Result |
|---|---:|
| Training rows | 20,868,210 |
| Training positives | 5,500 |
| Training prevalence | 0.026356% |
| Grid cells learned | 6,501 |
| Grid cells with ≥1 training fire day | 2,193 |
| Maximum historical grid risk | 3.3956% |

Validation results:

| Metric | Result |
|---|---:|
| Validation rows | 4,173,642 |
| Validation positives | 822 |
| Validation prevalence | 0.019695% |
| Average Precision / PR-AUC | 0.001118 |
| Random PR baseline | 0.000197 |
| Lift over random | 5.67× |
| ROC-AUC | 0.752907 |
| Brier score | 0.00019718 |

Top-risk capture:

| Highest-risk fraction | Fire grid-days captured | Capture rate | Precision |
|---|---:|---:|---:|
| Top 1% | 106 | 12.90% | 0.2540% |
| Top 5% | 262 | 31.87% | 0.1255% |
| Top 10% | 384 | 46.72% | 0.0920% |

### Interpretation

Historical location alone contains meaningful predictive information. The top 10% of spatially ranked grid-days contains nearly half of validation fire grid-days.

This baseline is therefore substantially more informative than a random ranking and establishes a meaningful benchmark for later models.

Generated outputs:

```text
outputs/tables/historical_spatial_risk_train_2005_2019.csv
outputs/tables/baseline_spatial_validation_metrics.csv
outputs/tables/baseline_spatial_top_risk_capture.csv
```

---

## 15. Baseline 2 — Historical Spatial + Seasonal Risk

Script:

```text
src/models/evaluate_spatiotemporal_baseline.py
```

### Method

This baseline learns historical risk for each:

```text
GRID_ID × calendar month
```

from 2005–2019.

Sparse cell-month estimates are smoothed toward the overall historical risk for that month using an empirical-Bayes-style prior with:

```text
PRIOR_OBSERVATIONS = 30
```

This baseline knows **where** fires have historically occurred and **when within the April–October season** fires have historically been more common, but it still uses no weather predictors.

### Historical monthly training rates

| Month | Positive grid-days | Historical rate |
|---|---:|---:|
| April | 376 | 0.01285% |
| May | 862 | 0.02852% |
| June | 1,153 | 0.03941% |
| July | 1,785 | 0.05905% |
| August | 998 | 0.03301% |
| September | 259 | 0.00885% |
| October | 67 | 0.00222% |

Additional training summary:

| Measurement | Result |
|---|---:|
| Grid-month groups | 45,507 |
| Groups with ≥1 historical fire | 3,506 |
| Maximum smoothed risk | 7.7108% |

### Validation results

| Metric | Result |
|---|---:|
| Validation rows | 4,173,642 |
| Validation positives | 822 |
| Validation prevalence | 0.019695% |
| Average Precision / PR-AUC | 0.001446 |
| Random PR baseline | 0.000197 |
| Lift over random | 7.34× |
| ROC-AUC | 0.796183 |
| Brier score | 0.00019775 |

Top-risk capture:

| Highest-risk fraction | Fire grid-days captured | Capture rate | Precision | Precision lift |
|---|---:|---:|---:|---:|
| Top 1% | 111 | 13.50% | 0.2660% | 13.50× |
| Top 5% | 267 | 32.48% | 0.1279% | 6.50× |
| Top 10% | 405 | 49.27% | 0.0970% | 4.93× |

### Improvement over spatial-only baseline

```text
Spatial-only PR-AUC:           0.001118
Spatial + seasonal PR-AUC:     0.001446
Absolute change:              +0.000328
Relative PR-AUC improvement:  +29.38%

Spatial-only ROC-AUC:           0.752907
Spatial + seasonal ROC-AUC:     0.796183
```

### Interpretation

Adding seasonality materially improves ranking performance. Historical timing within the fire season therefore contains useful information beyond location alone.

The Brier score did not improve, so the benefit at this stage is primarily ranking/discrimination rather than probability calibration. Calibration will be addressed explicitly later.

Generated outputs:

```text
outputs/tables/historical_spatiotemporal_risk_train_2005_2019.csv
outputs/tables/baseline_spatiotemporal_validation_metrics.csv
outputs/tables/baseline_spatiotemporal_top_risk_capture.csv
```

---

## 16. Current Benchmark to Beat

The current strongest validation benchmark is the historical spatial + seasonal baseline:

```text
PR-AUC:               0.001446
Lift over random:     7.34×
ROC-AUC:              0.796183
Top 10% fire capture: 49.27%
```

A weather-based model should be judged against this baseline, not merely against random chance or raw accuracy.

The next planned model is a weather-only logistic regression. Its purpose is to provide an interpretable first test of whether the engineered weather variables contribute predictive information beyond historical location/season patterns.

---

## 17. Current Modelling Roadmap

### Immediate next stage

1. Train weather-only logistic regression using 2005–2019.
2. Evaluate only on 2020–2022.
3. Compare PR-AUC, top-risk capture, ROC-AUC, and calibration with both historical baselines.
4. Inspect coefficient directions and feature scaling.
5. Address class imbalance using training-only methods such as class weighting where appropriate.

### Later modelling stages

Potential model families:

- Logistic regression.
- Random forest.
- Gradient-boosted decision trees.
- Additional nonlinear approaches only when they provide measurable validation improvement.

Potential later feature groups:

- Land cover.
- Vegetation / fuel indicators.
- Elevation, slope, and aspect.
- Distance to roads.
- Distance to populated places.
- Accessibility or recreation-related proxies.
- Lightning information, subject to data availability and leakage review.

Potential interpretation methods:

- Model coefficients for linear models.
- SHAP or comparable explanation methods for nonlinear models.
- Partial-dependence or accumulated-local-effect diagnostics when useful.

---

## 18. Exploratory Visualization Work

Visualization notebook:

```text
notebooks/07_linkedin_visuals.ipynb
```

Current visuals include:

- Annual wildfire counts, 2005–2025.
- Monthly wildfire seasonality.
- Fire-cause percentages.
- 10-km historical wildfire hotspot map.
- Human vs natural fire spatial patterns with selected population centres.
- Project workflow diagram.

These are communication / exploratory products and are kept separate from the production feature and modelling scripts.

The human-vs-natural spatial visualization suggests that human-caused records appear more concentrated in southern Manitoba and near populated corridors, whereas natural-caused records are more broadly distributed. This remains a visual association rather than a quantified predictor relationship and should not be presented as a causal finding.

---

## 19. Key Technical Decisions and Rationale

### Decision: Begin daily modelling in 2005

**Reason:** Earlier portions of the fire database contain substantially more records without exact usable dates. Daily prediction requires exact dates.

### Decision: Retain zero-size fires

**Reason:** A zero reported size may reflect reporting conventions or very small/extinguished fires rather than an invalid ignition. Removing them automatically could bias occurrence modelling.

### Decision: Exclude 16 near-border records from grid modelling

**Reason:** They are Manitoba-reported historical records but fall outside the official provincial modelling polygon. They remain in the audit dataset for transparency.

### Decision: Use a 10-km grid

**Reason:** Provides a tractable spatial unit for linking fire occurrences, gridded weather, and later environmental covariates while retaining meaningful spatial variation across Manitoba.

### Decision: Use Manitoba local dates

**Reason:** Fire occurrence is reported by calendar date locally. Aggregating UTC weather directly could assign late-evening or overnight weather to the wrong local day.

### Decision: Handle DST explicitly

**Reason:** Valid Manitoba local days can contain 23 or 25 hourly observations. Requiring exactly 24 would incorrectly reject valid source data.

### Decision: Match water-adjacent cells to nearest valid ERA5-Land location

**Reason:** ERA5-Land does not provide valid land variables at all nominal coordinates near major water bodies. Nearest-valid-land matching prevents systematic missing weather while retaining distance metadata for QA/QC.

### Decision: Download March as weather support

**Reason:** April 1 forecasts require antecedent rolling weather. March provides enough prior observations to compute complete 30-day features at the beginning of the modelling season.

### Decision: Predict t + 1 rather than same-day fire occurrence

**Reason:** Same-day weather may include observations occurring after an ignition has already happened. One-day-ahead prediction creates a clearer forecasting task and reduces leakage risk.

### Decision: Use a 1 mm rain threshold for `DAYS_SINCE_RAIN`

**Reason:** Very small trace precipitation should not necessarily reset a meaningful dry spell. The threshold is explicit and can later be sensitivity-tested.

### Decision: Split chronologically

**Reason:** Random row splitting would mix years and highly correlated neighboring observations. A chronological split better represents deployment to future fire seasons.

### Decision: Seal 2023–2025

**Reason:** The final test period should provide an unbiased estimate of generalization after all model and threshold decisions have been made using training and validation only.

### Decision: Establish historical baselines before weather ML

**Reason:** A useful wildfire model should outperform simple knowledge of where and when fires have historically occurred. Otherwise apparent ML performance may simply reflect spatial and seasonal priors.

### Decision: Do not use accuracy as the main metric

**Reason:** Positive wildfire grid-days occur in roughly 0.025% of observations. Accuracy would be dominated by the negative class.

---

## 20. Important Generated Tables and Outputs

### Data audit

```text
outputs/tables/multiyear_data_audit_2005_2025.csv
```

### Feature audit

```text
outputs/tables/next_day_feature_audit_2005_2025.csv
```

### Model split

```text
outputs/tables/model_split_manifest_2005_2025.csv
outputs/tables/model_split_summary_2005_2025.csv
```

### Spatial baseline

```text
outputs/tables/historical_spatial_risk_train_2005_2019.csv
outputs/tables/baseline_spatial_validation_metrics.csv
outputs/tables/baseline_spatial_top_risk_capture.csv
```

### Spatial + seasonal baseline

```text
outputs/tables/historical_spatiotemporal_risk_train_2005_2019.csv
outputs/tables/baseline_spatiotemporal_validation_metrics.csv
outputs/tables/baseline_spatiotemporal_top_risk_capture.csv
```

### Logs

Batch logs are stored locally under:

```text
outputs/logs/
```

They are excluded from version control.

---

## 21. Reproducibility Notes

Large source and processed datasets are intentionally not stored in GitHub.

The repository should preserve:

- Processing scripts.
- Feature scripts.
- Model scripts.
- Notebooks needed to explain exploratory/prototype stages.
- Documentation.
- Small summary tables/figures where appropriate.

The repository should not preserve:

- CDS API credentials.
- Large ERA5 NetCDF downloads.
- Large generated Parquet datasets.
- Temporary `.part` files.
- Local batch logs unless a specific log is intentionally needed for debugging documentation.

---

## 22. Known Limitations

1. Historical reporting practices can vary over time.
2. Some source fire coordinates may be approximate.
3. Fire occurrence records represent reported fires rather than a perfect census of every ignition.
4. The model currently predicts general wildfire occurrence rather than separate human-caused and natural-caused mechanisms.
5. ERA5-Land is coarser than station-scale meteorological observation.
6. Some water-adjacent Manitoba grid cells use a more distant valid ERA5-Land location.
7. Weather predictors alone do not encode vegetation, fuel availability, terrain, lightning, human access, or suppression behavior.
8. Historical spatial-risk features can encode both physical susceptibility and reporting/access patterns.
9. The positive class is extremely rare.
10. Validation results should not be interpreted as operational wildfire-warning performance.
11. The final 2023–2025 test evaluation has not yet been performed.

---

## 23. Responsible Use

This project is a research, educational, and portfolio prototype.

It is not an official wildfire forecast, emergency-warning product, government fire-danger replacement, or operational decision-support system.

Any future deployment would require independent scientific review, stronger predictor coverage, robust calibration, external validation, operational data engineering, monitoring, and collaboration with wildfire-domain specialists.

---

## 24. Current Checkpoint

At this checkpoint, the project has moved beyond data acquisition and pipeline construction.

Completed:

```text
Historical wildfire QA/QC              ✓
10-km spatial grid                     ✓
Daily fire targets                     ✓
ERA5-Land acquisition                  ✓
Daily weather processing               ✓
2005–2025 multi-year validation         ✓
Next-day weather feature engineering   ✓
21-year feature audit                  ✓
Chronological split                    ✓
Spatial historical baseline            ✓
Spatial + seasonal baseline            ✓
```

Next:

```text
Weather-only logistic regression
          ↓
Validation comparison against historical baselines
          ↓
Nonlinear models / additional features
          ↓
Model selection
          ↓
Final 2023–2025 test evaluation
```

The current validation benchmark to beat is the spatial + seasonal historical model with **PR-AUC 0.001446**, **ROC-AUC 0.796183**, and **49.27% capture of validation fire grid-days within the highest-risk 10% of grid-days**.
