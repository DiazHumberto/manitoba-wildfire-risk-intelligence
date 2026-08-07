"""Build next-day wildfire weather features for one Manitoba fire season."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_COLUMNS = [
    "TEMP_MAX_C",
    "TEMP_MIN_C",
    "TEMP_MEAN_C",
    "RH_MIN_PCT",
    "RH_MEAN_PCT",
    "WIND_MAX_MS",
    "WIND_MEAN_MS",
    "PRECIPITATION_MM",
]

ROLLING_FEATURE_COLUMNS = [
    "PRECIP_3D_MM",
    "PRECIP_7D_MM",
    "PRECIP_30D_MM",
    "TEMP_MAX_3D_MEAN_C",
    "TEMP_MAX_7D_MEAN_C",
    "RH_MIN_3D_MEAN_PCT",
    "RH_MIN_7D_MEAN_PCT",
    "WIND_MAX_3D_MEAN_MS",
    "WIND_MAX_7D_MEAN_MS",
    "DAYS_SINCE_RAIN",
]

RAIN_THRESHOLD_MM = 1.0
EXPECTED_GRID_CELLS = 6501


def find_project_root() -> Path:
    """Locate the repository root."""

    script_root = Path(__file__).resolve().parents[2]

    if (script_root / "data").exists():
        return script_root

    current = Path.cwd().resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "data").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate project root."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build next-day Manitoba wildfire weather "
            "features for one year."
        )
    )

    parser.add_argument(
        "year",
        type=int,
        help="Year to process, e.g. 2024.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing feature table.",
    )

    return parser.parse_args()


def validate_year(year: int) -> None:
    if year < 2005 or year > 2025:
        raise ValueError(
            "Year must be between 2005 and 2025."
        )


def add_rolling_features(
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Create antecedent weather features by grid cell."""

    print("\nCreating rolling precipitation features...")

    grouped_precip = weather.groupby(
        "GRID_ID",
        sort=False,
    )["PRECIPITATION_MM"]

    for window in (3, 7, 30):
        weather[
            f"PRECIP_{window}D_MM"
        ] = (
            grouped_precip
            .rolling(
                window=window,
                min_periods=window,
            )
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
        )

    print("Creating rolling temperature features...")

    for window in (3, 7):
        rolled = (
            weather
            .groupby(
                "GRID_ID",
                sort=False,
            )[["TEMP_MAX_C"]]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        weather[
            f"TEMP_MAX_{window}D_MEAN_C"
        ] = rolled["TEMP_MAX_C"]

    print("Creating rolling humidity features...")

    for window in (3, 7):
        rolled = (
            weather
            .groupby(
                "GRID_ID",
                sort=False,
            )[["RH_MIN_PCT"]]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        weather[
            f"RH_MIN_{window}D_MEAN_PCT"
        ] = rolled["RH_MIN_PCT"]

    print("Creating rolling wind features...")

    for window in (3, 7):
        rolled = (
            weather
            .groupby(
                "GRID_ID",
                sort=False,
            )[["WIND_MAX_MS"]]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        weather[
            f"WIND_MAX_{window}D_MEAN_MS"
        ] = rolled["WIND_MAX_MS"]

    print("Creating dry-spell feature...")

    # A day is considered a rain day when precipitation
    # reaches at least 1 mm.
    is_rain_day = (
        weather["PRECIPITATION_MM"]
        >= RAIN_THRESHOLD_MM
    )

    rain_event_number = (
        is_rain_day
        .groupby(
            weather["GRID_ID"],
            sort=False,
        )
        .cumsum()
    )

    dry_day = (
        ~is_rain_day
    ).astype("int16")

    weather[
        "DAYS_SINCE_RAIN"
    ] = (
        dry_day
        .groupby(
            [
                weather["GRID_ID"],
                rain_event_number,
            ],
            sort=False,
        )
        .cumsum()
        .astype("int16")
    )

    return weather


def build_features(
    year: int,
    project_root: Path,
) -> pd.DataFrame:
    """Build the next-day feature table."""

    interim_folder = (
        project_root
        / "data"
        / "interim"
    )

    weather_path = (
        interim_folder
        / f"era5_grid_weather_{year}_march_october.parquet"
    )

    targets_path = (
        interim_folder
        / "manitoba_daily_fire_targets_2005_2025.parquet"
    )

    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather table not found: {weather_path}"
        )

    if not targets_path.exists():
        raise FileNotFoundError(
            f"Fire target table not found: {targets_path}"
        )

    print("=" * 72)
    print("NEXT-DAY WEATHER FEATURE ENGINEERING")
    print("=" * 72)
    print("Year:", year)
    print("Weather:", weather_path.name)

    # ----------------------------------------------------------
    # Read weather support table
    # ----------------------------------------------------------

    weather = pd.read_parquet(
        weather_path,
        columns=[
            "LOCAL_DATE",
            "GRID_ID",
            *WEATHER_COLUMNS,
        ],
    )

    weather["LOCAL_DATE"] = pd.to_datetime(
        weather["LOCAL_DATE"]
    ).dt.normalize()

    weather = (
        weather
        .sort_values(
            [
                "GRID_ID",
                "LOCAL_DATE",
            ]
        )
        .reset_index(drop=True)
    )

    print("\nWeather rows:", len(weather))
    print(
        "Weather dates:",
        weather["LOCAL_DATE"].nunique(),
    )
    print(
        "Grid cells:",
        weather["GRID_ID"].nunique(),
    )

    # ----------------------------------------------------------
    # Basic input validation
    # ----------------------------------------------------------

    expected_support_dates = pd.date_range(
        start=f"{year}-03-01",
        end=f"{year}-10-31",
        freq="D",
    )

    expected_weather_rows = (
        len(expected_support_dates)
        * EXPECTED_GRID_CELLS
    )

    if len(weather) != expected_weather_rows:
        raise RuntimeError(
            "Unexpected number of weather rows: "
            f"{len(weather):,} != "
            f"{expected_weather_rows:,}"
        )

    if weather["GRID_ID"].nunique() != EXPECTED_GRID_CELLS:
        raise RuntimeError(
            "Unexpected number of grid cells."
        )

    duplicate_weather = int(
        weather.duplicated(
            [
                "GRID_ID",
                "LOCAL_DATE",
            ]
        ).sum()
    )

    if duplicate_weather:
        raise RuntimeError(
            f"Weather table contains {duplicate_weather} duplicates."
        )

    missing_weather = int(
        weather[
            WEATHER_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_weather:
        raise RuntimeError(
            f"Weather table contains {missing_weather} missing values."
        )

    # ----------------------------------------------------------
    # Feature engineering
    # ----------------------------------------------------------

    weather = add_rolling_features(
        weather
    )

    # Predictor dates:
    # March 31 through October 30.
    #
    # Their forecast dates are:
    # April 1 through October 31.
    predictor_start = pd.Timestamp(
        f"{year}-03-31"
    )

    predictor_end = pd.Timestamp(
        f"{year}-10-30"
    )

    features = weather.loc[
        weather["LOCAL_DATE"].between(
            predictor_start,
            predictor_end,
        )
    ].copy()

    features = features.rename(
        columns={
            "LOCAL_DATE":
                "PREDICTOR_DATE",
        }
    )

    features[
        "FORECAST_DATE"
    ] = (
        features["PREDICTOR_DATE"]
        + pd.Timedelta(days=1)
    )

    # ----------------------------------------------------------
    # Seasonality features
    # ----------------------------------------------------------

    features[
        "DAY_OF_YEAR"
    ] = (
        features[
            "FORECAST_DATE"
        ]
        .dt.dayofyear
        .astype("int16")
    )

    features[
        "DOY_SIN"
    ] = np.sin(
        2
        * np.pi
        * features["DAY_OF_YEAR"]
        / 365.25
    )

    features[
        "DOY_COS"
    ] = np.cos(
        2
        * np.pi
        * features["DAY_OF_YEAR"]
        / 365.25
    )

    features[
        "YEAR"
    ] = year

    # ----------------------------------------------------------
    # Attach NEXT-DAY wildfire target
    # ----------------------------------------------------------

    targets = pd.read_parquet(
        targets_path,
        columns=[
            "FIRE_DATE",
            "GRID_ID",
            "FIRE_OCCURRED",
            "FIRE_COUNT",
        ],
    )

    targets[
        "FIRE_DATE"
    ] = pd.to_datetime(
        targets["FIRE_DATE"]
    ).dt.normalize()

    targets = targets.loc[
        targets["FIRE_DATE"].between(
            f"{year}-04-01",
            f"{year}-10-31",
        )
    ].copy()

    expected_positive_days = len(
        targets
    )

    expected_fire_count = int(
        targets[
            "FIRE_COUNT"
        ].sum()
    )

    targets = targets.rename(
        columns={
            "FIRE_DATE":
                "FORECAST_DATE",
            "FIRE_OCCURRED":
                "FIRE_OCCURRED_NEXT_DAY",
            "FIRE_COUNT":
                "FIRE_COUNT_NEXT_DAY",
        }
    )

    features = features.merge(
        targets[
            [
                "FORECAST_DATE",
                "GRID_ID",
                "FIRE_OCCURRED_NEXT_DAY",
                "FIRE_COUNT_NEXT_DAY",
            ]
        ],
        how="left",
        on=[
            "FORECAST_DATE",
            "GRID_ID",
        ],
        validate="one_to_one",
    )

    features[
        "FIRE_OCCURRED_NEXT_DAY"
    ] = (
        features[
            "FIRE_OCCURRED_NEXT_DAY"
        ]
        .fillna(0)
        .astype("int8")
    )

    features[
        "FIRE_COUNT_NEXT_DAY"
    ] = (
        features[
            "FIRE_COUNT_NEXT_DAY"
        ]
        .fillna(0)
        .astype("int16")
    )

    # ----------------------------------------------------------
    # Final validation
    # ----------------------------------------------------------

    feature_columns = (
        WEATHER_COLUMNS
        + ROLLING_FEATURE_COLUMNS
        + [
            "DAY_OF_YEAR",
            "DOY_SIN",
            "DOY_COS",
        ]
    )

    expected_forecast_dates = pd.date_range(
        start=f"{year}-04-01",
        end=f"{year}-10-31",
        freq="D",
    )

    expected_rows = (
        len(expected_forecast_dates)
        * EXPECTED_GRID_CELLS
    )

    duplicate_rows = int(
        features.duplicated(
            [
                "FORECAST_DATE",
                "GRID_ID",
            ]
        ).sum()
    )

    missing_features = int(
        features[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    positive_days = int(
        features[
            "FIRE_OCCURRED_NEXT_DAY"
        ].sum()
    )

    represented_fires = int(
        features[
            "FIRE_COUNT_NEXT_DAY"
        ].sum()
    )

    print("\n" + "=" * 72)
    print("FEATURE TABLE VALIDATION")
    print("=" * 72)

    print(
        "Rows:",
        f"{len(features):,}",
    )

    print(
        "Expected rows:",
        f"{expected_rows:,}",
    )

    print(
        "Forecast dates:",
        features[
            "FORECAST_DATE"
        ].nunique(),
    )

    print(
        "Forecast range:",
        features[
            "FORECAST_DATE"
        ].min().date(),
        "to",
        features[
            "FORECAST_DATE"
        ].max().date(),
    )

    print(
        "Grid cells:",
        features[
            "GRID_ID"
        ].nunique(),
    )

    print(
        "Duplicate date-grid rows:",
        duplicate_rows,
    )

    print(
        "Missing feature values:",
        missing_features,
    )

    print(
        "Positive next-day grid-cell targets:",
        positive_days,
    )

    print(
        "Expected positive targets:",
        expected_positive_days,
    )

    print(
        "Individual fires represented:",
        represented_fires,
    )

    print(
        "Expected individual fires:",
        expected_fire_count,
    )

    checks = {
        "row_count":
            len(features)
            == expected_rows,

        "forecast_dates":
            features[
                "FORECAST_DATE"
            ].nunique()
            == len(expected_forecast_dates),

        "grid_cells":
            features[
                "GRID_ID"
            ].nunique()
            == EXPECTED_GRID_CELLS,

        "duplicates":
            duplicate_rows == 0,

        "missing_features":
            missing_features == 0,

        "positive_targets":
            positive_days
            == expected_positive_days,

        "fire_count":
            represented_fires
            == expected_fire_count,

        "forecast_start":
            features[
                "FORECAST_DATE"
            ].min()
            == pd.Timestamp(
                f"{year}-04-01"
            ),

        "forecast_end":
            features[
                "FORECAST_DATE"
            ].max()
            == pd.Timestamp(
                f"{year}-10-31"
            ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            "Feature validation failed: "
            + ", ".join(
                failed_checks
            )
        )

    # ----------------------------------------------------------
    # Reduce numeric storage
    # ----------------------------------------------------------

    float_columns = (
        WEATHER_COLUMNS
        + [
            col
            for col
            in ROLLING_FEATURE_COLUMNS
            if col != "DAYS_SINCE_RAIN"
        ]
        + [
            "DOY_SIN",
            "DOY_COS",
        ]
    )

    features[
        float_columns
    ] = features[
        float_columns
    ].astype("float32")

    features[
        "DAYS_SINCE_RAIN"
    ] = features[
        "DAYS_SINCE_RAIN"
    ].astype("int16")

    features[
        "YEAR"
    ] = features[
        "YEAR"
    ].astype("int16")

    # ----------------------------------------------------------
    # Clean column order
    # ----------------------------------------------------------

    features = features[
        [
            "PREDICTOR_DATE",
            "FORECAST_DATE",
            "YEAR",
            "GRID_ID",

            *WEATHER_COLUMNS,

            "PRECIP_3D_MM",
            "PRECIP_7D_MM",
            "PRECIP_30D_MM",

            "TEMP_MAX_3D_MEAN_C",
            "TEMP_MAX_7D_MEAN_C",

            "RH_MIN_3D_MEAN_PCT",
            "RH_MIN_7D_MEAN_PCT",

            "WIND_MAX_3D_MEAN_MS",
            "WIND_MAX_7D_MEAN_MS",

            "DAYS_SINCE_RAIN",

            "DAY_OF_YEAR",
            "DOY_SIN",
            "DOY_COS",

            "FIRE_OCCURRED_NEXT_DAY",
            "FIRE_COUNT_NEXT_DAY",
        ]
    ]

    return features


def main() -> None:
    args = parse_arguments()

    year = args.year

    validate_year(year)

    project_root = find_project_root()

    output_folder = (
        project_root
        / "data"
        / "processed"
        / "features"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_folder
        / f"manitoba_next_day_features_{year}.parquet"
    )

    if (
        output_path.exists()
        and not args.overwrite
    ):
        print(
            "Output already exists:",
            output_path,
        )
        print(
            "Use --overwrite to rebuild it."
        )
        return

    features = build_features(
        year=year,
        project_root=project_root,
    )

    temporary_path = Path(
        str(output_path)
        + ".tmp"
    )

    features.to_parquet(
        temporary_path,
        index=False,
    )

    # Confirm that the written Parquet can be reopened.
    written = pd.read_parquet(
        temporary_path,
        columns=[
            "FORECAST_DATE",
            "GRID_ID",
            "FIRE_OCCURRED_NEXT_DAY",
            "FIRE_COUNT_NEXT_DAY",
        ],
    )

    if len(written) != len(features):
        temporary_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Written Parquet row count does not match feature table."
        )

    temporary_path.replace(
        output_path
    )

    print("\n" + "=" * 72)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 72)

    print(
        "Saved:",
        output_path,
    )

    print(
        "Rows:",
        f"{len(features):,}",
    )


if __name__ == "__main__":
    main()
