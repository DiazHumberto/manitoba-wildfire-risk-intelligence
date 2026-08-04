"""Download and validate ERA5-Land data for one Manitoba fire season.

The download window includes:
- March through October: complete monthly files
- November 1: UTC buffer needed to complete October 31 local time

March is retained to support lagged and rolling weather features for April.
"""

from __future__ import annotations

import argparse
import calendar
import time
from pathlib import Path

import cdsapi
import xarray as xr


DATASET_NAME = "reanalysis-era5-land"

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]

EXPECTED_NETCDF_VARIABLES = {
    "t2m",
    "d2m",
    "u10",
    "v10",
    "tp",
}

TIMES = [
    f"{hour:02d}:00"
    for hour in range(24)
]

MANITOBA_AREA = [
    60.2,      # North
    -102.3,    # West
    48.8,      # South
    -89.5,     # East
]

# March supports rolling features; April–October is the fire season.
DOWNLOAD_MONTHS = list(range(3, 11))

MINIMUM_FILE_SIZE_BYTES = 1_000_000


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
) -> list[str]:
    """Return every day number for a calendar month."""

    number_of_days = calendar.monthrange(
        year,
        month,
    )[1]

    return [
        f"{day:02d}"
        for day in range(
            1,
            number_of_days + 1,
        )
    ]


def create_request(
    *,
    year: int,
    month: int,
    days: list[str],
) -> dict:
    """Create one ERA5-Land request."""

    return {
        "variable": VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": MANITOBA_AREA,
    }


def validate_netcdf(
    path: Path,
    *,
    expected_hours: int,
) -> tuple[bool, str]:
    """Check file size, variables, timestamps, and readability."""

    if not path.exists():
        return False, "file does not exist"

    file_size = path.stat().st_size

    if file_size < MINIMUM_FILE_SIZE_BYTES:
        return False, (
            f"file is too small "
            f"({file_size / 1_000_000:.2f} MB)"
        )

    try:
        with xr.open_dataset(
            path,
            engine="netcdf4",
        ) as dataset:
            time_name = next(
                (
                    name
                    for name in [
                        "valid_time",
                        "time",
                    ]
                    if (
                        name in dataset.coords
                        or name in dataset.dims
                    )
                ),
                None,
            )

            if time_name is None:
                return False, "time coordinate was not found"

            actual_hours = int(
                dataset[time_name].size
            )

            if actual_hours != expected_hours:
                return False, (
                    f"expected {expected_hours} hours, "
                    f"found {actual_hours}"
                )

            available_variables = set(
                dataset.data_vars
            )

            missing_variables = sorted(
                EXPECTED_NETCDF_VARIABLES
                - available_variables
            )

            if missing_variables:
                return False, (
                    "missing variables: "
                    + ", ".join(missing_variables)
                )

    except Exception as error:
        return False, (
            f"could not open NetCDF: "
            f"{type(error).__name__}: {error}"
        )

    return True, (
        f"valid NetCDF with "
        f"{expected_hours} hourly timestamps"
    )


def download_file(
    *,
    client: cdsapi.Client,
    request: dict,
    output_path: Path,
    expected_hours: int,
    retries: int,
    force: bool,
) -> None:
    """Download one file safely and validate it."""

    if output_path.exists() and not force:
        valid, message = validate_netcdf(
            output_path,
            expected_hours=expected_hours,
        )

        if valid:
            size_mb = (
                output_path.stat().st_size
                / 1_000_000
            )

            print(
                f"SKIP: {output_path.name} "
                f"({size_mb:.2f} MB) — {message}"
            )

            return

        print(
            f"INVALID: {output_path.name} — {message}"
        )
        print("Removing invalid existing file.")
        output_path.unlink()

    elif output_path.exists() and force:
        print(
            f"FORCE: removing {output_path.name}"
        )
        output_path.unlink()

    temporary_path = output_path.with_name(
        output_path.name + ".part"
    )

    if temporary_path.exists():
        print(
            f"Removing incomplete temporary file: "
            f"{temporary_path.name}"
        )
        temporary_path.unlink()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for attempt in range(1, retries + 1):
        print(
            f"\nDownloading: {output_path.name}"
        )
        print(
            f"Attempt: {attempt}/{retries}"
        )

        try:
            client.retrieve(
                DATASET_NAME,
                request,
                str(temporary_path),
            )

            valid, message = validate_netcdf(
                temporary_path,
                expected_hours=expected_hours,
            )

            if not valid:
                raise RuntimeError(
                    f"Downloaded file failed validation: "
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
                f"COMPLETED: {output_path.name}"
            )
            print(
                f"Size: {size_mb:.2f} MB"
            )
            print(
                f"Validation: {message}"
            )

            return

        except Exception as error:
            print(
                f"Attempt failed: "
                f"{type(error).__name__}: {error}"
            )

            if temporary_path.exists():
                temporary_path.unlink()

            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after "
                    f"{retries} attempts: "
                    f"{output_path.name}"
                ) from error

            wait_seconds = 10 * attempt

            print(
                f"Retrying in "
                f"{wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)


def build_download_jobs(
    *,
    year: int,
    weather_folder: Path,
) -> list[dict]:
    """Create monthly and buffer download jobs."""

    jobs = []

    for month in DOWNLOAD_MONTHS:
        days = get_month_days(
            year,
            month,
        )

        output_path = (
            weather_folder
            / (
                "era5_land_manitoba_"
                f"{year}_{month:02d}_hourly.nc"
            )
        )

        jobs.append(
            {
                "label": (
                    f"{year}-{month:02d}"
                ),
                "request": create_request(
                    year=year,
                    month=month,
                    days=days,
                ),
                "output_path": output_path,
                "expected_hours": (
                    len(days) * 24
                ),
            }
        )

    november_buffer_path = (
        weather_folder
        / (
            "era5_land_manitoba_"
            f"{year}_11_01_hourly.nc"
        )
    )

    jobs.append(
        {
            "label": f"{year}-11-01 buffer",
            "request": create_request(
                year=year,
                month=11,
                days=["01"],
            ),
            "output_path": (
                november_buffer_path
            ),
            "expected_hours": 24,
        }
    )

    return jobs


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download and validate one year of "
            "ERA5-Land Manitoba weather data."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help=(
            "Year to download. "
            "The project period is 2005–2025."
        ),
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help=(
            "Maximum download attempts per file. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Redownload files even when they "
            "already pass validation."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the downloader."""

    arguments = parse_arguments()

    if not 2005 <= arguments.year <= 2025:
        raise ValueError(
            "Year must be between 2005 and 2025."
        )

    if arguments.retries < 1:
        raise ValueError(
            "Retries must be at least 1."
        )

    project_root = find_project_root()

    weather_folder = (
        project_root
        / "data"
        / "raw"
        / "weather"
        / "era5_land"
        / str(arguments.year)
    )

    weather_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("ERA5-Land Manitoba downloader")
    print("=" * 70)
    print("Project root:", project_root)
    print("Year:", arguments.year)
    print("Output folder:", weather_folder)
    print(
        "Monthly window:",
        "March through October",
    )
    print(
        "Additional buffer:",
        "November 1",
    )
    print(
        "Force download:",
        arguments.force,
    )

    jobs = build_download_jobs(
        year=arguments.year,
        weather_folder=weather_folder,
    )

    client = cdsapi.Client()

    for job_number, job in enumerate(
        jobs,
        start=1,
    ):
        print("\n" + "-" * 70)

        print(
            f"Job {job_number}/{len(jobs)}: "
            f"{job['label']}"
        )

        download_file(
            client=client,
            request=job["request"],
            output_path=job["output_path"],
            expected_hours=job[
                "expected_hours"
            ],
            retries=arguments.retries,
            force=arguments.force,
        )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    valid_files = 0

    for job in jobs:
        path = job["output_path"]

        valid, message = validate_netcdf(
            path,
            expected_hours=job[
                "expected_hours"
            ],
        )

        status = (
            "VALID"
            if valid
            else "INVALID"
        )

        size_mb = (
            path.stat().st_size / 1_000_000
            if path.exists()
            else 0
        )

        print(
            f"{status}: {path.name} "
            f"({size_mb:.2f} MB) — {message}"
        )

        if valid:
            valid_files += 1

    print("\nFiles expected:", len(jobs))
    print("Files valid:", valid_files)
    print(
        "Files invalid:",
        len(jobs) - valid_files,
    )

    if valid_files != len(jobs):
        raise RuntimeError(
            "One or more files failed final validation."
        )

    print(
        f"\nYear {arguments.year} "
        f"is ready for processing."
    )


if __name__ == "__main__":
    main()
