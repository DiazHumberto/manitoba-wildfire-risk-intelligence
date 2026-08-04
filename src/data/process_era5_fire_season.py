"""Process one year of Manitoba ERA5-Land wildfire-season data.

Inputs
------
- Monthly hourly ERA5-Land files from March through October
- November 1 hourly buffer
- Validated Manitoba grid-to-ERA5 matching table
- Daily Manitoba wildfire target table

Outputs
-------
- March through October grid-cell weather table
- April through October weather-and-fire model table

March is retained to support 3-day, 7-day, and 30-day rolling
weather features for dates at the beginning of April.
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr


TIMEZONE = "America/Winnipeg"

RAW_VARIABLES = {
    "t2m",
    "d2m",
    "u10",
    "v10",
    "tp",
}

WEATHER_VARIABLES = [
    "TEMP_MAX_C",
    "TEMP_MIN_C",
    "TEMP_MEAN_C",
    "RH_MIN_PCT",
    "RH_MEAN_PCT",
    "WIND_MAX_MS",
    "WIND_MEAN_MS",
    "PRECIPITATION_MM",
]

TARGET_FIELDS = [
    "FIRE_DATE",
    "GRID_ID",
    "FIRE_OCCURRED",
    "FIRE_COUNT",
    "NATURAL_FIRE_COUNT",
    "HUMAN_FIRE_COUNT",
    "UNKNOWN_FIRE_COUNT",
    "TOTAL_BURNED_HA",
    "MAX_FIRE_SIZE_HA",
]

COUNT_COLUMNS = [
    "FIRE_OCCURRED",
    "FIRE_COUNT",
    "NATURAL_FIRE_COUNT",
    "HUMAN_FIRE_COUNT",
    "UNKNOWN_FIRE_COUNT",
]

SIZE_COLUMNS = [
    "TOTAL_BURNED_HA",
    "MAX_FIRE_SIZE_HA",
]

MATCH_METADATA_COLUMNS = [
    "GRID_ID",
    "CENTER_LATITUDE",
    "CENTER_LONGITUDE",
    "MB_AREA_KM2",
    "MB_COVERAGE_PCT",
    "ERA5_MATCH_DISTANCE_KM",
    "ERA5_MATCH_QUALITY",
]


def find_project_root() -> Path:
    """Find the repository root containing the data directory."""

    script_root = Path(__file__).resolve().parents[2]

    if (script_root / "data").exists():
        return script_root

    current = Path.cwd().resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "data").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the project root containing the data folder."
    )


def get_month_days(
    year: int,
    month: int,
) -> int:
    """Return the number of calendar days in a month."""

    return calendar.monthrange(
        year,
        month,
    )[1]


def build_hourly_paths(
    *,
    weather_folder: Path,
    year: int,
) -> list[Path]:
    """Return the required hourly files in chronological order."""

    paths = []

    for month in range(3, 11):
        paths.append(
            weather_folder
            / (
                "era5_land_manitoba_"
                f"{year}_{month:02d}_hourly.nc"
            )
        )

    paths.append(
        weather_folder
        / (
            "era5_land_manitoba_"
            f"{year}_11_01_hourly.nc"
        )
    )

    return paths


def validate_hourly_inputs(
    *,
    paths: list[Path],
    year: int,
) -> None:
    """Validate all required hourly NetCDF inputs."""

    expected_hours_by_file = {}

    for month in range(3, 11):
        filename = (
            "era5_land_manitoba_"
            f"{year}_{month:02d}_hourly.nc"
        )

        expected_hours_by_file[filename] = (
            get_month_days(year, month) * 24
        )

    buffer_filename = (
        "era5_land_manitoba_"
        f"{year}_11_01_hourly.nc"
    )

    expected_hours_by_file[buffer_filename] = 24

    validation_errors = []

    for path in paths:
        if not path.exists():
            validation_errors.append(
                f"{path.name}: file does not exist"
            )
            continue

        try:
            with xr.open_dataset(
                path,
                engine="netcdf4",
            ) as dataset:
                time_name = next(
                    (
                        candidate
                        for candidate in [
                            "valid_time",
                            "time",
                        ]
                        if (
                            candidate in dataset.coords
                            or candidate in dataset.dims
                        )
                    ),
                    None,
                )

                if time_name is None:
                    validation_errors.append(
                        f"{path.name}: no time coordinate"
                    )
                    continue

                expected_hours = (
                    expected_hours_by_file[path.name]
                )

                actual_hours = int(
                    dataset[time_name].size
                )

                if actual_hours != expected_hours:
                    validation_errors.append(
                        f"{path.name}: expected "
                        f"{expected_hours} hours, found "
                        f"{actual_hours}"
                    )

                available_variables = set(
                    dataset.data_vars
                )

                missing_variables = sorted(
                    RAW_VARIABLES
                    - available_variables
                )

                if missing_variables:
                    validation_errors.append(
                        f"{path.name}: missing variables "
                        + ", ".join(missing_variables)
                    )

        except Exception as error:
            validation_errors.append(
                f"{path.name}: "
                f"{type(error).__name__}: {error}"
            )

    if validation_errors:
        details = "\n".join(
            f"- {message}"
            for message in validation_errors
        )

        raise RuntimeError(
            "Hourly input validation failed:\n"
            f"{details}"
        )

    print(
        f"Validated {len(paths)} hourly input files."
    )


def validate_parquet(
    *,
    path: Path,
    expected_rows: int,
    required_columns: set[str],
) -> tuple[bool, str]:
    """Validate a Parquet file using its metadata and schema."""

    if not path.exists():
        return False, "file does not exist"

    try:
        parquet_file = pq.ParquetFile(path)

        actual_rows = (
            parquet_file.metadata.num_rows
        )

        if actual_rows != expected_rows:
            return False, (
                f"expected {expected_rows} rows, "
                f"found {actual_rows}"
            )

        available_columns = set(
            parquet_file.schema.names
        )

        missing_columns = sorted(
            required_columns
            - available_columns
        )

        if missing_columns:
            return False, (
                "missing columns: "
                + ", ".join(missing_columns)
            )

    except Exception as error:
        return False, (
            f"{type(error).__name__}: {error}"
        )

    return True, (
        f"valid Parquet with {expected_rows} rows"
    )


def write_parquet_safely(
    *,
    dataframe: pd.DataFrame,
    output_path: Path,
    expected_rows: int,
    required_columns: set[str],
) -> None:
    """Write a Parquet file through a temporary validated file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name + ".part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    dataframe.to_parquet(
        temporary_path,
        index=False,
    )

    valid, message = validate_parquet(
        path=temporary_path,
        expected_rows=expected_rows,
        required_columns=required_columns,
    )

    if not valid:
        temporary_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"Temporary output failed validation: "
            f"{message}"
        )

    temporary_path.replace(
        output_path
    )

    size_mb = (
        output_path.stat().st_size
        / 1_000_000
    )

    print(
        f"Saved: {output_path.name}"
    )
    print(
        f"Size: {size_mb:.2f} MB"
    )
    print(
        f"Validation: {message}"
    )


def create_daily_weather(
    hourly_weather_raw: xr.Dataset,
) -> tuple[
    xr.Dataset,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    """Convert hourly ERA5-Land variables into daily weather."""

    if "valid_time" not in hourly_weather_raw.coords:
        if "time" in hourly_weather_raw.coords:
            hourly_weather_raw = (
                hourly_weather_raw.rename(
                    {"time": "valid_time"}
                )
            )
        else:
            raise KeyError(
                "No valid_time or time coordinate was found."
            )

    hourly_weather_raw = (
        hourly_weather_raw
        .sortby("valid_time")
    )

    time_index = pd.DatetimeIndex(
        hourly_weather_raw[
            "valid_time"
        ].values
    )

    if time_index.duplicated().any():
        duplicate_count = int(
            time_index.duplicated().sum()
        )

        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "hourly timestamps."
        )

    hourly_weather = xr.Dataset(
        {
            "TEMPERATURE_C": (
                hourly_weather_raw["t2m"]
                - 273.15
            ),
            "DEWPOINT_C": (
                hourly_weather_raw["d2m"]
                - 273.15
            ),
            "WIND_SPEED_MS": np.sqrt(
                hourly_weather_raw["u10"] ** 2
                + hourly_weather_raw["v10"] ** 2
            ),
            "PRECIPITATION_MM": (
                hourly_weather_raw["tp"]
                * 1000
            ),
        }
    )

    hourly_weather[
        "RELATIVE_HUMIDITY_PCT"
    ] = (
        100
        * np.exp(
            (
                17.625
                * hourly_weather["DEWPOINT_C"]
                / (
                    243.04
                    + hourly_weather["DEWPOINT_C"]
                )
            )
            -
            (
                17.625
                * hourly_weather["TEMPERATURE_C"]
                / (
                    243.04
                    + hourly_weather["TEMPERATURE_C"]
                )
            )
        )
    ).clip(
        min=0,
        max=100,
    )

    utc_times = pd.DatetimeIndex(
        hourly_weather[
            "valid_time"
        ].values
    ).tz_localize("UTC")

    local_times = utc_times.tz_convert(
        TIMEZONE
    )

    instantaneous_dates = (
        local_times
        .tz_localize(None)
        .normalize()
    )

    precipitation_dates = (
        (
            local_times
            - pd.Timedelta(
                nanoseconds=1
            )
        )
        .tz_localize(None)
        .normalize()
    )

    hourly_weather = (
        hourly_weather.assign_coords(
            LOCAL_DATE=(
                "valid_time",
                instantaneous_dates.to_numpy(
                    dtype="datetime64[ns]"
                ),
            )
        )
    )

    precipitation_hourly = (
        hourly_weather[
            "PRECIPITATION_MM"
        ]
        .assign_coords(
            PRECIPITATION_DATE=(
                "valid_time",
                precipitation_dates.to_numpy(
                    dtype="datetime64[ns]"
                ),
            )
        )
    )

    daily_precipitation = (
        precipitation_hourly
        .groupby(
            "PRECIPITATION_DATE"
        )
        .sum("valid_time")
        .rename(
            {
                "PRECIPITATION_DATE":
                "LOCAL_DATE"
            }
        )
    )

    daily_weather = xr.Dataset(
        {
            "TEMP_MAX_C": (
                hourly_weather[
                    "TEMPERATURE_C"
                ]
                .groupby("LOCAL_DATE")
                .max("valid_time")
            ),
            "TEMP_MIN_C": (
                hourly_weather[
                    "TEMPERATURE_C"
                ]
                .groupby("LOCAL_DATE")
                .min("valid_time")
            ),
            "TEMP_MEAN_C": (
                hourly_weather[
                    "TEMPERATURE_C"
                ]
                .groupby("LOCAL_DATE")
                .mean("valid_time")
            ),
            "RH_MIN_PCT": (
                hourly_weather[
                    "RELATIVE_HUMIDITY_PCT"
                ]
                .groupby("LOCAL_DATE")
                .min("valid_time")
            ),
            "RH_MEAN_PCT": (
                hourly_weather[
                    "RELATIVE_HUMIDITY_PCT"
                ]
                .groupby("LOCAL_DATE")
                .mean("valid_time")
            ),
            "WIND_MAX_MS": (
                hourly_weather[
                    "WIND_SPEED_MS"
                ]
                .groupby("LOCAL_DATE")
                .max("valid_time")
            ),
            "WIND_MEAN_MS": (
                hourly_weather[
                    "WIND_SPEED_MS"
                ]
                .groupby("LOCAL_DATE")
                .mean("valid_time")
            ),
            "PRECIPITATION_MM": (
                daily_precipitation
            ),
        }
    )

    return (
        daily_weather,
        instantaneous_dates,
        precipitation_dates,
    )


def get_expected_local_day_hours(
    date: pd.Timestamp,
) -> int:
    """Return the actual number of hours in a Manitoba local day."""

    date = pd.Timestamp(date).normalize()

    local_start = date.tz_localize(
        TIMEZONE
    )

    local_end = (
        date
        + pd.Timedelta(days=1)
    ).tz_localize(
        TIMEZONE
    )

    utc_duration = (
        local_end.tz_convert("UTC")
        - local_start.tz_convert("UTC")
    )

    return int(
        utc_duration
        / pd.Timedelta(hours=1)
    )


def validate_complete_dates(
    *,
    expected_dates: pd.DatetimeIndex,
    instantaneous_dates: pd.DatetimeIndex,
    precipitation_dates: pd.DatetimeIndex,
) -> None:
    """Confirm each local date has its expected DST-adjusted hours."""

    instantaneous_counts = pd.Series(
        instantaneous_dates
    ).value_counts()

    precipitation_counts = pd.Series(
        precipitation_dates
    ).value_counts()

    incomplete_dates = []
    dst_adjusted_dates = []

    for date in expected_dates:
        expected_hours = (
            get_expected_local_day_hours(
                date
            )
        )

        instantaneous_hours = int(
            instantaneous_counts.get(
                date,
                0,
            )
        )

        precipitation_hours = int(
            precipitation_counts.get(
                date,
                0,
            )
        )

        if expected_hours != 24:
            dst_adjusted_dates.append(
                {
                    "LOCAL_DATE": date,
                    "EXPECTED_HOURS":
                        expected_hours,
                }
            )

        if (
            instantaneous_hours
            != expected_hours
            or precipitation_hours
            != expected_hours
        ):
            incomplete_dates.append(
                {
                    "LOCAL_DATE": date,
                    "EXPECTED_HOURS":
                        expected_hours,
                    "INSTANTANEOUS_HOURS":
                        instantaneous_hours,
                    "PRECIPITATION_HOURS":
                        precipitation_hours,
                }
            )

    if incomplete_dates:
        incomplete_table = pd.DataFrame(
            incomplete_dates
        )

        raise RuntimeError(
            "One or more local dates are incomplete:\n"
            f"{incomplete_table.to_string(index=False)}"
        )

    print(
        f"Validated {len(expected_dates)} "
        "complete local dates."
    )

    if dst_adjusted_dates:
        print(
            "DST-adjusted local dates:"
        )

        for record in dst_adjusted_dates:
            date_text = (
                pd.Timestamp(
                    record["LOCAL_DATE"]
                )
                .strftime("%Y-%m-%d")
            )

            print(
                f"- {date_text}: "
                f"{record['EXPECTED_HOURS']} hours"
            )


def assign_weather_to_grid(
    *,
    daily_weather: xr.Dataset,
    era5_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each Manitoba grid cell to its ERA5-Land point."""

    grid_ids = (
        era5_matches["GRID_ID"]
        .to_numpy()
    )

    latitude_indexer = xr.DataArray(
        era5_matches[
            "ERA5_LATITUDE"
        ].to_numpy(),
        dims="GRID_ID",
        coords={
            "GRID_ID": grid_ids
        },
    )

    longitude_indexer = xr.DataArray(
        era5_matches[
            "ERA5_LONGITUDE"
        ].to_numpy(),
        dims="GRID_ID",
        coords={
            "GRID_ID": grid_ids
        },
    )

    weather_at_grid = daily_weather.sel(
        latitude=latitude_indexer,
        longitude=longitude_indexer,
        method="nearest",
    )

    weather_grid_df = (
        weather_at_grid
        .to_dataframe()
        .reset_index()
        .rename(
            columns={
                "latitude":
                    "ERA5_LATITUDE",
                "longitude":
                    "ERA5_LONGITUDE",
            }
        )
    )

    weather_grid_df = (
        weather_grid_df.drop(
            columns=[
                "number",
                "expver",
            ],
            errors="ignore",
        )
    )

    weather_grid_df["LOCAL_DATE"] = (
        pd.to_datetime(
            weather_grid_df[
                "LOCAL_DATE"
            ]
        )
        .dt.normalize()
    )

    weather_grid_df = (
        weather_grid_df.merge(
            era5_matches[
                MATCH_METADATA_COLUMNS
            ],
            on="GRID_ID",
            how="left",
            validate="many_to_one",
        )
    )

    for column in WEATHER_VARIABLES:
        weather_grid_df[column] = (
            weather_grid_df[column]
            .astype("float32")
        )

    float_metadata_columns = [
        "ERA5_LATITUDE",
        "ERA5_LONGITUDE",
        "CENTER_LATITUDE",
        "CENTER_LONGITUDE",
        "MB_AREA_KM2",
        "MB_COVERAGE_PCT",
        "ERA5_MATCH_DISTANCE_KM",
    ]

    for column in float_metadata_columns:
        if column in weather_grid_df:
            weather_grid_df[column] = (
                weather_grid_df[column]
                .astype("float32")
            )

    weather_grid_df = (
        weather_grid_df.sort_values(
            [
                "LOCAL_DATE",
                "GRID_ID",
            ]
        )
        .reset_index(drop=True)
    )

    return weather_grid_df


def build_model_table(
    *,
    weather_grid_df: pd.DataFrame,
    fire_targets: pd.DataFrame,
    fire_season_start: pd.Timestamp,
    fire_season_end: pd.Timestamp,
) -> tuple[pd.DataFrame, int, int]:
    """Combine April–October weather with wildfire targets."""

    fire_season_weather = (
        weather_grid_df.loc[
            weather_grid_df[
                "LOCAL_DATE"
            ].between(
                fire_season_start,
                fire_season_end,
            )
        ]
        .copy()
    )

    fire_season_targets = (
        fire_targets.loc[
            fire_targets[
                "FIRE_DATE"
            ].between(
                fire_season_start,
                fire_season_end,
            ),
            TARGET_FIELDS,
        ]
        .copy()
    )

    expected_positive_rows = len(
        fire_season_targets
    )

    expected_fire_count = int(
        fire_season_targets[
            "FIRE_COUNT"
        ].sum()
    )

    model_table = (
        fire_season_weather.merge(
            fire_season_targets,
            left_on=[
                "LOCAL_DATE",
                "GRID_ID",
            ],
            right_on=[
                "FIRE_DATE",
                "GRID_ID",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    model_table[COUNT_COLUMNS] = (
        model_table[COUNT_COLUMNS]
        .fillna(0)
        .astype("int16")
    )

    model_table[SIZE_COLUMNS] = (
        model_table[SIZE_COLUMNS]
        .fillna(0.0)
        .astype("float32")
    )

    model_table = (
        model_table
        .drop(
            columns="FIRE_DATE"
        )
        .sort_values(
            [
                "LOCAL_DATE",
                "GRID_ID",
            ]
        )
        .reset_index(drop=True)
    )

    actual_positive_rows = int(
        model_table[
            "FIRE_OCCURRED"
        ].sum()
    )

    actual_fire_count = int(
        model_table[
            "FIRE_COUNT"
        ].sum()
    )

    if (
        actual_positive_rows
        != expected_positive_rows
    ):
        raise RuntimeError(
            "Positive target rows were not "
            "preserved during the merge."
        )

    if actual_fire_count != expected_fire_count:
        raise RuntimeError(
            "Individual fire counts were not "
            "preserved during the merge."
        )

    return (
        model_table,
        actual_positive_rows,
        actual_fire_count,
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Process one year of Manitoba "
            "ERA5-Land wildfire-season data."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help=(
            "Year to process. "
            "The project period is 2005–2025."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild outputs even when existing "
            "files pass validation."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run annual ERA5-Land processing."""

    arguments = parse_arguments()

    if not 2005 <= arguments.year <= 2025:
        raise ValueError(
            "Year must be between 2005 and 2025."
        )

    year = arguments.year

    project_root = find_project_root()

    weather_folder = (
        project_root
        / "data"
        / "raw"
        / "weather"
        / "era5_land"
        / str(year)
    )

    interim_folder = (
        project_root
        / "data"
        / "interim"
    )

    era5_match_path = (
        interim_folder
        / "manitoba_grid_to_era5_land_matches.parquet"
    )

    targets_path = (
        interim_folder
        / "manitoba_daily_fire_targets_2005_2025.parquet"
    )

    weather_output_path = (
        interim_folder
        / (
            "era5_grid_weather_"
            f"{year}_march_october.parquet"
        )
    )

    model_output_path = (
        interim_folder
        / (
            "manitoba_model_table_"
            f"{year}_fire_season.parquet"
        )
    )

    if not era5_match_path.exists():
        raise FileNotFoundError(
            f"ERA5 matching table not found: "
            f"{era5_match_path}"
        )

    if not targets_path.exists():
        raise FileNotFoundError(
            f"Fire target table not found: "
            f"{targets_path}"
        )

    era5_matches = pd.read_parquet(
        era5_match_path
    )

    if era5_matches[
        "GRID_ID"
    ].duplicated().any():
        raise ValueError(
            "ERA5 matching table contains "
            "duplicate GRID_ID values."
        )

    grid_count = len(
        era5_matches
    )

    support_dates = pd.date_range(
        start=f"{year}-03-01",
        end=f"{year}-10-31",
        freq="D",
    )

    fire_season_dates = pd.date_range(
        start=f"{year}-04-01",
        end=f"{year}-10-31",
        freq="D",
    )

    expected_weather_rows = (
        len(support_dates)
        * grid_count
    )

    expected_model_rows = (
        len(fire_season_dates)
        * grid_count
    )

    weather_required_columns = {
        "LOCAL_DATE",
        "GRID_ID",
        *WEATHER_VARIABLES,
    }

    model_required_columns = {
        *weather_required_columns,
        "FIRE_OCCURRED",
        "FIRE_COUNT",
    }

    weather_valid, weather_message = (
        validate_parquet(
            path=weather_output_path,
            expected_rows=expected_weather_rows,
            required_columns=weather_required_columns,
        )
    )

    model_valid, model_message = (
        validate_parquet(
            path=model_output_path,
            expected_rows=expected_model_rows,
            required_columns=model_required_columns,
        )
    )

    print("=" * 70)
    print("ERA5-Land annual processor")
    print("=" * 70)
    print("Project root:", project_root)
    print("Year:", year)
    print("Grid cells:", grid_count)
    print(
        "Support dates:",
        len(support_dates),
        "(March–October)",
    )
    print(
        "Fire-season dates:",
        len(fire_season_dates),
        "(April–October)",
    )
    print(
        "Expected weather rows:",
        expected_weather_rows,
    )
    print(
        "Expected model rows:",
        expected_model_rows,
    )

    if (
        weather_valid
        and model_valid
        and not arguments.force
    ):
        print(
            "\nSKIP: annual outputs already "
            "pass validation."
        )
        print(
            weather_output_path.name,
            "—",
            weather_message,
        )
        print(
            model_output_path.name,
            "—",
            model_message,
        )
        return

    hourly_paths = build_hourly_paths(
        weather_folder=weather_folder,
        year=year,
    )

    validate_hourly_inputs(
        paths=hourly_paths,
        year=year,
    )

    print("\nOpening hourly files...")

    hourly_weather_raw = xr.open_mfdataset(
        hourly_paths,
        combine="by_coords",
        chunks="auto",
        engine="netcdf4",
    )

    if "valid_time" not in hourly_weather_raw.coords:
        if "time" in hourly_weather_raw.coords:
            hourly_weather_raw = (
                hourly_weather_raw.rename(
                    {"time": "valid_time"}
                )
            )

    hourly_weather_raw = (
        hourly_weather_raw.sortby(
            "valid_time"
        )
    )

    hourly_time_index = pd.DatetimeIndex(
        hourly_weather_raw[
            "valid_time"
        ].values
    )

    expected_hourly_count = (
        sum(
            get_month_days(year, month)
            for month in range(3, 11)
        )
        + 1
    ) * 24

    print(
        "Hourly timestamps:",
        len(hourly_time_index),
    )
    print(
        "Expected hourly timestamps:",
        expected_hourly_count,
    )
    print(
        "Duplicate hourly timestamps:",
        int(
            hourly_time_index
            .duplicated()
            .sum()
        ),
    )

    if len(hourly_time_index) != expected_hourly_count:
        raise RuntimeError(
            "Unexpected total hourly timestamp count."
        )

    print("\nCreating daily weather...")

    (
        daily_weather,
        instantaneous_dates,
        precipitation_dates,
    ) = create_daily_weather(
        hourly_weather_raw
    )

    validate_complete_dates(
        expected_dates=support_dates,
        instantaneous_dates=instantaneous_dates,
        precipitation_dates=precipitation_dates,
    )

    support_date_values = (
        support_dates.to_numpy(
            dtype="datetime64[ns]"
        )
    )

    daily_weather_support = (
        daily_weather.sel(
            LOCAL_DATE=support_date_values
        )
        .load()
    )

    hourly_weather_raw.close()

    print(
        "Daily weather dimensions:",
        dict(
            daily_weather_support.sizes
        ),
    )

    print(
        "\nAssigning weather to Manitoba grid cells..."
    )

    weather_grid_df = assign_weather_to_grid(
        daily_weather=daily_weather_support,
        era5_matches=era5_matches,
    )

    if len(weather_grid_df) != expected_weather_rows:
        raise RuntimeError(
            "Unexpected weather-table row count."
        )

    if weather_grid_df.duplicated(
        subset=[
            "LOCAL_DATE",
            "GRID_ID",
        ]
    ).any():
        raise RuntimeError(
            "Weather table contains duplicate "
            "date-grid rows."
        )

    missing_weather_values = int(
        weather_grid_df[
            WEATHER_VARIABLES
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Weather-table rows:",
        len(weather_grid_df),
    )
    print(
        "Missing weather values:",
        missing_weather_values,
    )

    if missing_weather_values != 0:
        raise RuntimeError(
            "Weather table contains missing values."
        )

    fire_targets = pd.read_parquet(
        targets_path
    )

    fire_targets["FIRE_DATE"] = (
        pd.to_datetime(
            fire_targets[
                "FIRE_DATE"
            ]
        )
        .dt.normalize()
    )

    (
        model_table,
        positive_grid_cell_days,
        individual_fires,
    ) = build_model_table(
        weather_grid_df=weather_grid_df,
        fire_targets=fire_targets,
        fire_season_start=pd.Timestamp(
            f"{year}-04-01"
        ),
        fire_season_end=pd.Timestamp(
            f"{year}-10-31"
        ),
    )

    if len(model_table) != expected_model_rows:
        raise RuntimeError(
            "Unexpected model-table row count."
        )

    if model_table.duplicated(
        subset=[
            "LOCAL_DATE",
            "GRID_ID",
        ]
    ).any():
        raise RuntimeError(
            "Model table contains duplicate "
            "date-grid rows."
        )

    print(
        "\nModel-table rows:",
        len(model_table),
    )
    print(
        "Positive grid-cell days:",
        positive_grid_cell_days,
    )
    print(
        "Individual fires represented:",
        individual_fires,
    )

    print("\nWriting annual outputs...")

    write_parquet_safely(
        dataframe=weather_grid_df,
        output_path=weather_output_path,
        expected_rows=expected_weather_rows,
        required_columns=weather_required_columns,
    )

    write_parquet_safely(
        dataframe=model_table,
        output_path=model_output_path,
        expected_rows=expected_model_rows,
        required_columns=model_required_columns,
    )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    weather_valid, weather_message = (
        validate_parquet(
            path=weather_output_path,
            expected_rows=expected_weather_rows,
            required_columns=weather_required_columns,
        )
    )

    model_valid, model_message = (
        validate_parquet(
            path=model_output_path,
            expected_rows=expected_model_rows,
            required_columns=model_required_columns,
        )
    )

    print(
        "Weather output:",
        "VALID"
        if weather_valid
        else "INVALID",
        "—",
        weather_message,
    )

    print(
        "Model output:",
        "VALID"
        if model_valid
        else "INVALID",
        "—",
        model_message,
    )

    if not weather_valid or not model_valid:
        raise RuntimeError(
            "One or more annual outputs "
            "failed final validation."
        )

    print(
        f"\nYear {year} processing completed."
    )


if __name__ == "__main__":
    main()
