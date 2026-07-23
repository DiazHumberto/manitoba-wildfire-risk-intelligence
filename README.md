# Manitoba Wildfire Risk Intelligence

**Status:** Active development  
**Study area:** Manitoba, Canada  
**Current modelling period:** 2005–2025  
**Spatial resolution:** 10 km × 10 km grid cells  
**Primary tools:** Python, GeoPandas, Xarray, ERA5-Land, Scikit-learn

## Project Overview

**Manitoba Wildfire Risk Intelligence** is a geospatial data science and machine-learning project designed to estimate daily wildfire ignition risk across Manitoba.

The project combines historical wildfire records with gridded weather data and, in later stages, environmental and human-accessibility variables. The final goal is to produce an explainable model that estimates the probability of a new wildfire occurring in each Manitoba grid cell during the following day.

The project is being developed as a research and portfolio prototype. It is not intended to replace official wildfire monitoring, forecasting, or emergency-warning systems.

## Research Question

> Can explainable machine learning predict the daily probability of a new wildfire occurring across Manitoba more effectively than weather-based and fire-weather baselines alone?

## Why This Project Matters

Wildfire risk is influenced by several interacting factors, including:

- Temperature
- Relative humidity
- Wind speed
- Precipitation and recent dry conditions
- Vegetation and fuel characteristics
- Seasonal patterns
- Historical fire activity
- Human access and infrastructure
- Lightning and natural ignition patterns

A spatially explicit risk model can demonstrate how environmental data, GIS, and artificial intelligence may be combined to support wildfire preparedness and decision-making.

## Planned Prediction Task

The intended modelling unit is one **grid cell and day**.

Each observation will eventually represent:

```text
Weather and environmental conditions on day t
                       ↓
Probability of a new wildfire on day t + 1
```

The planned target is:

```text
1 = At least one new wildfire occurs in the grid cell the following day
0 = No new wildfire occurs in the grid cell the following day
```

The target will be shifted forward by one day before model training. This prevents the model from using information observed after a fire has already started.

## Current Project Scope

| Component | Current decision |
|---|---|
| Study area | Manitoba, Canada |
| Historical modelling period | 2005–2025 |
| Spatial unit | 10 km × 10 km grid cell |
| Number of Manitoba grid cells | 6,501 |
| Temporal unit | Daily |
| Forecast horizon | One day ahead |
| Initial fire-season focus | April–October |
| Main model type | Binary classification |
| Primary evaluation focus | Precision-recall, calibration, and high-risk-area recall |

## Data Sources

### Canadian National Fire Database

Historical wildfire point records are obtained from the Canadian National Fire Database maintained through the Canadian Wildland Fire Information System.

Relevant fields include:

- Fire identifier
- Reporting agency
- Reported date
- Latitude and longitude
- Fire size in hectares
- Fire cause
- Fire type

### Manitoba Provincial Boundary

The official Manitoba provincial boundary is used to:

- Validate wildfire coordinates
- Define the study area
- Create the 10-km analysis grid
- Exclude records outside the provincial modelling boundary

### ERA5-Land

Hourly ERA5-Land reanalysis data are used to create daily gridded weather features.

Current weather variables include:

- 2-metre air temperature
- 2-metre dew-point temperature
- 10-metre eastward wind component
- 10-metre northward wind component
- Total precipitation

Derived daily variables include:

- Maximum temperature
- Minimum temperature
- Mean temperature
- Minimum relative humidity
- Mean relative humidity
- Maximum wind speed
- Mean wind speed
- Total daily precipitation

Weather timestamps are converted from UTC to Manitoba local time before daily aggregation.

## Data Processing Workflow

```text
Canadian wildfire records
          ↓
Filter to Manitoba
          ↓
Audit dates, coordinates, sizes, causes, and duplicates
          ↓
Retain records from 2005–2025
          ↓
Create a 10-km Manitoba grid
          ↓
Assign each wildfire to a grid cell
          ↓
Aggregate fires by date and grid cell
          ↓
Download hourly ERA5-Land weather
          ↓
Convert UTC observations to Manitoba local dates
          ↓
Create daily weather features
          ↓
Match each Manitoba cell to a valid ERA5-Land location
          ↓
Combine weather and wildfire targets
          ↓
Create lagged and rolling features
          ↓
Train and evaluate wildfire-risk models
```

## Current Progress

### Completed

- [x] Initialized the project repository and Python environment
- [x] Downloaded the Canadian National Fire Database point records
- [x] Filtered historical records to Manitoba
- [x] Audited dates, coordinates, fire sizes, causes, and duplicates
- [x] Established 2005–2025 as the modelling period
- [x] Removed exact duplicate records
- [x] Retained zero-size fires with a quality-control flag
- [x] Validated fire locations against the Manitoba boundary
- [x] Created a 10 km × 10 km Manitoba grid
- [x] Assigned eligible wildfires to grid cells
- [x] Created the daily positive-fire target table
- [x] Configured access to the Copernicus Climate Data Store
- [x] Developed and validated an ERA5-Land weather pilot
- [x] Corrected weather matches for coastal and water-adjacent cells
- [x] Processed the complete July 2024 weather dataset
- [x] Combined July 2024 weather and wildfire targets

### In Progress

- [ ] Process the complete April–October 2024 fire season
- [ ] Generalize the ERA5-Land workflow for multiple years
- [ ] Process weather data for 2005–2025

### Planned

- [ ] Calculate lagged weather variables
- [ ] Calculate 3-day, 7-day, and 30-day precipitation totals
- [ ] Calculate consecutive dry-day indicators
- [ ] Add seasonal variables
- [ ] Shift wildfire targets to the following day
- [ ] Add vegetation, land-cover, terrain, and accessibility variables
- [ ] Build baseline models
- [ ] Train machine-learning models
- [ ] Evaluate temporal and geographic generalization
- [ ] Add explainable AI using SHAP
- [ ] Create an interactive wildfire-risk dashboard

## Current Data Summary

### Historical Fire Records

The original Manitoba subset contains:

| Measurement | Result |
|---|---:|
| Manitoba records, 1959–2025 | 28,797 |
| Records with an exact usable date | 25,940 |
| Records without an exact date | 2,857 |
| Records in the modelling period, 2005–2025 | 7,933 |
| Exact duplicate rows removed | 2 |
| Cleaned modelling-period records | 7,931 |
| Records inside the Manitoba boundary | 7,915 |
| Near-border records retained for historical auditing | 16 |
| Zero-size fire records retained | 153 |

Most records without exact dates occur during 2000–2003. For this reason, the daily modelling period begins in 2005.

### Fire Causes, 2005–2025

| Cause | Fire records |
|---|---:|
| Natural | 4,661 |
| Human | 3,255 |
| Unknown | 17 |

Fire cause is used for exploratory analysis but will not be used as an input feature for general fire-occurrence prediction because it is normally known only after a fire is recorded.

### Spatial Grid

| Measurement | Result |
|---|---:|
| Grid size | 10 km × 10 km |
| Cells intersecting Manitoba | 6,501 |
| Cells containing at least one historical fire | 2,593 |
| Maximum historical fires in one cell | 140 |
| Eligible fires assigned to a grid cell | 7,915 |
| Unmatched fires | 0 |

### Daily Fire Targets

| Measurement | Result |
|---|---:|
| Positive grid-cell days | 7,356 |
| Grid-cell days with multiple fires | 434 |
| Maximum fires in one cell on one day | 10 |
| Duplicate date-grid combinations | 0 |

A positive grid-cell day represents one date and grid-cell combination with at least one recorded wildfire.

### ERA5-Land Weather Validation

The first weather pilot covered two complete Manitoba-local days in July 2024.

| Measurement | Result |
|---|---:|
| Manitoba grid cells | 6,501 |
| Pilot dates | 2 |
| Expected grid-cell days | 13,002 |
| Actual grid-cell days | 13,002 |
| Unique ERA5-Land locations used | 6,006 |
| Missing weather values after correction | 0 |
| Maximum nearest-land matching distance | 35.03 km |
| Positive pilot grid-cell days | 3 |

Cells near Hudson Bay, large lakes, or water-dominated boundaries may require a more distant valid ERA5-Land point. Match distance and match-quality fields are preserved for quality control.

### July 2024 Processing

| Measurement | Result |
|---|---:|
| Complete local dates | 31 |
| Manitoba grid cells | 6,501 |
| Weather and target rows | 201,531 |
| Missing weather values | 0 |
| Positive grid-cell days | 111 |
| Individual fires represented | 118 |

## Planned Models

The project will begin with simple, interpretable baselines before adding more complex models.

Planned comparisons include:

1. Seasonal historical-risk baseline
2. Weather-only logistic regression
3. Random forest
4. Gradient-boosted decision trees
5. A full model containing weather, spatial, vegetation, terrain, and human-accessibility features

A more complex model will only be considered valuable if it provides measurable improvement over simpler baselines.

## Evaluation Strategy

Wildfire occurrence is a highly imbalanced classification problem because most grid cells do not experience a new fire on most days.

For this reason, ordinary accuracy will not be treated as the main metric.

Planned evaluation metrics include:

- Precision-recall area under the curve
- Recall among the highest-risk grid cells
- Precision
- Recall
- F1 score
- Brier score
- Probability calibration
- False alarms per detected fire
- Confusion matrices at operationally meaningful thresholds

The data will be divided chronologically rather than using a random train-test split. This better represents the real task of predicting future wildfire seasons.

Geographic holdout tests may also be used to evaluate whether the model generalizes to regions that were not represented during training.

## Planned Feature Engineering

Potential weather and temporal features include:

- Previous-day temperature
- Previous-day relative humidity
- Previous-day wind speed
- Previous-day precipitation
- Three-day precipitation
- Seven-day precipitation
- Thirty-day precipitation
- Consecutive dry days
- Day of year
- Month
- Seasonal sine and cosine transformations

Potential spatial features include:

- Historical fire density
- Elevation
- Slope
- Aspect
- Land-cover class
- Forest or fuel type
- Vegetation indices
- Distance to roads
- Distance to populated places
- Distance to recreational or accessible areas

All features will be reviewed carefully for data leakage.

## Repository Structure

```text
manitoba-wildfire-risk-intelligence/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── fires/
│   │   ├── boundaries/
│   │   ├── weather/
│   │   ├── fire_weather/
│   │   └── environmental/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_fire_data_exploration.ipynb
│   ├── 02_manitoba_grid.ipynb
│   ├── 03_daily_fire_targets.ipynb
│   ├── 04_era5_weather_pilot.ipynb
│   ├── 05_era5_weather_july_2024.ipynb
│   └── 06_era5_weather_2024_fire_season.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
│
├── models/
├── outputs/
│   ├── figures/
│   ├── maps/
│   └── tables/
│
└── app/
```

## Notebook Descriptions

### `01_fire_data_exploration.ipynb`

- Loads the Canadian National Fire Database
- Filters records to Manitoba
- Examines dates, coordinates, sizes, and causes
- Removes exact duplicates
- Validates locations against the Manitoba boundary
- Saves the cleaned wildfire dataset

### `02_manitoba_grid.ipynb`

- Projects Manitoba into a metre-based coordinate system
- Creates a regular 10-km grid
- Calculates provincial coverage within border cells
- Assigns historical fires to grid cells
- Creates historical fire-density summaries

### `03_daily_fire_targets.ipynb`

- Aggregates fires by date and grid cell
- Creates the positive daily target table
- Preserves natural, human, and unknown fire counts
- Creates the Manitoba grid-centre reference table

### `04_era5_weather_pilot.ipynb`

- Tests ERA5-Land downloads
- Converts temperature and precipitation units
- Calculates wind speed and relative humidity
- Converts UTC timestamps to Manitoba local time
- Creates daily weather summaries
- Matches weather to all Manitoba grid cells
- Resolves missing land-weather locations

### `05_era5_weather_july_2024.ipynb`

- Processes the complete July 2024 hourly weather dataset
- Adds an August buffer for time-zone completeness
- Creates 31 complete daily weather records
- Builds the first full monthly weather-target table

### `06_era5_weather_2024_fire_season.ipynb`

- Expands the validated workflow to the full 2024 fire season
- Processes weather from April through October
- Prepares the workflow for multi-year automation

## Installation

Clone the repository:

```bash
git clone https://github.com/DiazHumberto/manitoba-wildfire-risk-intelligence.git
cd manitoba-wildfire-risk-intelligence
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project currently uses Python packages including:

- pandas
- NumPy
- GeoPandas
- Shapely
- PyProj
- Matplotlib
- Xarray
- Dask
- NetCDF4
- cdsapi
- Scikit-learn
- PyArrow

## Data Availability

Large raw and processed datasets are not committed to GitHub.

The following folders are excluded through `.gitignore`:

```text
data/raw/
data/interim/
data/processed/
models/
selected generated outputs
```

Users wishing to reproduce the complete workflow must download the original datasets from their official providers and configure their own Copernicus Climate Data Store account.

Personal API credentials are never stored in the repository.

## Limitations

Current limitations include:

- Historical fire reporting practices vary by year and agency.
- Exact dates are unavailable for some older fire records.
- Some historical fire coordinates may represent approximate reporting locations.
- A small number of Manitoba-reported fires fall slightly outside the official provincial boundary.
- ERA5-Land has a coarser resolution than local weather-station measurements.
- Coastal and water-adjacent cells may be matched to a more distant valid land point.
- Fire occurrence is strongly imbalanced across space and time.
- Historical records indicate reported fires, not necessarily every ignition.
- Weather alone cannot explain all human-caused or lightning-caused ignitions.
- The project has not yet produced or validated a predictive model.

These limitations will be documented throughout the modelling and evaluation stages.

## Responsible Use

This project is an educational, research, and portfolio prototype.

It must not be used as:

- An official wildfire forecast
- An emergency-warning system
- A replacement for government fire-danger products
- A basis for operational deployment without independent validation
- A substitute for expert wildfire-management decisions

Official wildfire information and emergency instructions should always be obtained from the appropriate federal, provincial, and local authorities.

## Author

**Humberto Eleazar Díaz Maridueña**

Environmental professional with a Bachelor of Environmental Engineering and a Post-Degree Diploma in Artificial Intelligence and Machine Learning.

Based in Winnipeg, Manitoba, Canada.

- Portfolio: https://diazhumberto.github.io/
- GitHub: https://github.com/DiazHumberto
- LinkedIn: https://www.linkedin.com/in/humberto-e-diaz

## Project Status

This repository is under active development.

The current focus is building a reproducible multi-year weather and wildfire data pipeline. Predictive modelling, explainable AI, and an interactive risk dashboard will be added after the data-processing workflow has been fully validated.
