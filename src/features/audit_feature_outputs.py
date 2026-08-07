"""Audit annual next-day wildfire feature tables for 2005–2025."""

from pathlib import Path

import pandas as pd


YEARS = range(2005, 2026)

EXPECTED_GRID_CELLS = 6501
EXPECTED_FORECAST_DAYS = 214
EXPECTED_ROWS_PER_YEAR = (
    EXPECTED_GRID_CELLS
    * EXPECTED_FORECAST_DAYS
)

FEATURE_COLUMNS = [
    "TEMP_MAX_C",
    "TEMP_MIN_C",
    "TEMP_MEAN_C",
    "RH_MIN_PCT",
    "RH_MEAN_PCT",
    "WIND_MAX_MS",
    "WIND_MEAN_MS",
    "PRECIPITATION_MM",
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
]


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


def main() -> None:
    project_root = find_project_root()

    feature_folder = (
        project_root
        / "data"
        / "processed"
        / "features"
    )

    targets_path = (
        project_root
        / "data"
        / "interim"
        / "manitoba_daily_fire_targets_2005_2025.parquet"
    )

    output_folder = (
        project_root
        / "outputs"
        / "tables"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Original wildfire targets
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

    targets["FIRE_DATE"] = pd.to_datetime(
        targets["FIRE_DATE"]
    ).dt.normalize()

    print("=" * 78)
    print("MANITOBA WILDFIRE RISK INTELLIGENCE")
    print("NEXT-DAY FEATURE TABLE AUDIT: 2005–2025")
    print("=" * 78)

    audit_rows = []

    # ----------------------------------------------------------
    # Audit each annual feature table
    # ----------------------------------------------------------

    for year in YEARS:

        print("\n" + "-" * 78)
        print("AUDITING YEAR:", year)

        feature_path = (
            feature_folder
            / f"manitoba_next_day_features_{year}.parquet"
        )

        if not feature_path.exists():

            print("FAIL — feature file missing.")

            audit_rows.append(
                {
                    "YEAR": year,
                    "STATUS": "FAIL",
                    "FAILED_CHECKS": "missing_file",
                }
            )

            continue

        features = pd.read_parquet(
            feature_path
        )

        # ------------------------------------------------------
        # Dates
        # ------------------------------------------------------

        features["PREDICTOR_DATE"] = pd.to_datetime(
            features["PREDICTOR_DATE"]
        ).dt.normalize()

        features["FORECAST_DATE"] = pd.to_datetime(
            features["FORECAST_DATE"]
        ).dt.normalize()

        predictor_start = features[
            "PREDICTOR_DATE"
        ].min()

        predictor_end = features[
            "PREDICTOR_DATE"
        ].max()

        forecast_start = features[
            "FORECAST_DATE"
        ].min()

        forecast_end = features[
            "FORECAST_DATE"
        ].max()

        forecast_dates = features[
            "FORECAST_DATE"
        ].nunique()

        grid_cells = features[
            "GRID_ID"
        ].nunique()

        # ------------------------------------------------------
        # Structural checks
        # ------------------------------------------------------

        row_count = len(features)

        duplicate_rows = int(
            features.duplicated(
                subset=[
                    "FORECAST_DATE",
                    "GRID_ID",
                ]
            ).sum()
        )

        missing_features = int(
            features[
                FEATURE_COLUMNS
            ]
            .isna()
            .sum()
            .sum()
        )

        # Predictor date must always be exactly one day
        # before forecast date.
        date_offset_errors = int(
            (
                features["FORECAST_DATE"]
                - features["PREDICTOR_DATE"]
                != pd.Timedelta(days=1)
            ).sum()
        )

        # ------------------------------------------------------
        # Target checks
        # ------------------------------------------------------

        positive_targets = int(
            features[
                "FIRE_OCCURRED_NEXT_DAY"
            ].sum()
        )

        individual_fires = int(
            features[
                "FIRE_COUNT_NEXT_DAY"
            ].sum()
        )

        year_targets = targets.loc[
            targets["FIRE_DATE"].between(
                f"{year}-04-01",
                f"{year}-10-31",
            )
        ].copy()

        expected_positive_targets = len(
            year_targets
        )

        expected_individual_fires = int(
            year_targets[
                "FIRE_COUNT"
            ].sum()
        )

        # ------------------------------------------------------
        # Target consistency
        # ------------------------------------------------------

        invalid_binary_targets = int(
            (
                ~features[
                    "FIRE_OCCURRED_NEXT_DAY"
                ].isin([0, 1])
            ).sum()
        )

        target_count_mismatches = int(
            (
                (
                    features[
                        "FIRE_OCCURRED_NEXT_DAY"
                    ] == 0
                )
                &
                (
                    features[
                        "FIRE_COUNT_NEXT_DAY"
                    ] != 0
                )
            ).sum()
        )

        # ------------------------------------------------------
        # Year field
        # ------------------------------------------------------

        invalid_year_rows = int(
            (
                features["YEAR"]
                != year
            ).sum()
        )

        # ------------------------------------------------------
        # Basic feature sanity
        # ------------------------------------------------------

        negative_dry_days = int(
            (
                features[
                    "DAYS_SINCE_RAIN"
                ] < 0
            ).sum()
        )

        invalid_day_of_year = int(
            (
                features[
                    "DAY_OF_YEAR"
                ]
                != features[
                    "FORECAST_DATE"
                ].dt.dayofyear
            ).sum()
        )

        # ------------------------------------------------------
        # Expected dates
        # ------------------------------------------------------

        expected_predictor_start = pd.Timestamp(
            f"{year}-03-31"
        )

        expected_predictor_end = pd.Timestamp(
            f"{year}-10-30"
        )

        expected_forecast_start = pd.Timestamp(
            f"{year}-04-01"
        )

        expected_forecast_end = pd.Timestamp(
            f"{year}-10-31"
        )

        # ------------------------------------------------------
        # Validation dictionary
        # ------------------------------------------------------

        checks = {
            "row_count":
                row_count
                == EXPECTED_ROWS_PER_YEAR,

            "forecast_dates":
                forecast_dates
                == EXPECTED_FORECAST_DAYS,

            "grid_cells":
                grid_cells
                == EXPECTED_GRID_CELLS,

            "duplicates":
                duplicate_rows == 0,

            "missing_features":
                missing_features == 0,

            "date_offset":
                date_offset_errors == 0,

            "predictor_start":
                predictor_start
                == expected_predictor_start,

            "predictor_end":
                predictor_end
                == expected_predictor_end,

            "forecast_start":
                forecast_start
                == expected_forecast_start,

            "forecast_end":
                forecast_end
                == expected_forecast_end,

            "positive_targets":
                positive_targets
                == expected_positive_targets,

            "individual_fires":
                individual_fires
                == expected_individual_fires,

            "binary_targets":
                invalid_binary_targets == 0,

            "target_count_consistency":
                target_count_mismatches == 0,

            "year_field":
                invalid_year_rows == 0,

            "dry_days":
                negative_dry_days == 0,

            "day_of_year":
                invalid_day_of_year == 0,
        }

        failed_checks = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        status = (
            "PASS"
            if not failed_checks
            else "FAIL"
        )

        file_size_mb = (
            feature_path.stat().st_size
            / 1_000_000
        )

        audit_rows.append(
            {
                "YEAR":
                    year,

                "STATUS":
                    status,

                "ROWS":
                    row_count,

                "FORECAST_DATES":
                    forecast_dates,

                "GRID_CELLS":
                    grid_cells,

                "DUPLICATES":
                    duplicate_rows,

                "MISSING_FEATURE_VALUES":
                    missing_features,

                "DATE_OFFSET_ERRORS":
                    date_offset_errors,

                "POSITIVE_TARGETS":
                    positive_targets,

                "EXPECTED_POSITIVE_TARGETS":
                    expected_positive_targets,

                "INDIVIDUAL_FIRES":
                    individual_fires,

                "EXPECTED_INDIVIDUAL_FIRES":
                    expected_individual_fires,

                "FILE_SIZE_MB":
                    round(
                        file_size_mb,
                        2,
                    ),

                "FAILED_CHECKS":
                    ", ".join(
                        failed_checks
                    ),
            }
        )

        print("Status:", status)

        print(
            "Rows:",
            f"{row_count:,}",
        )

        print(
            "Forecast dates:",
            forecast_dates,
        )

        print(
            "Grid cells:",
            grid_cells,
        )

        print(
            "Duplicate rows:",
            duplicate_rows,
        )

        print(
            "Missing feature values:",
            missing_features,
        )

        print(
            "Date-offset errors:",
            date_offset_errors,
        )

        print(
            "Positive targets:",
            positive_targets,
        )

        print(
            "Individual fires:",
            individual_fires,
        )

        if failed_checks:

            print(
                "Failed checks:",
                failed_checks,
            )

        del features

    # ----------------------------------------------------------
    # Final multi-year summary
    # ----------------------------------------------------------

    audit = pd.DataFrame(
        audit_rows
    )

    audit_path = (
        output_folder
        / "next_day_feature_audit_2005_2025.csv"
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    print("\n" + "=" * 78)
    print("FINAL FEATURE AUDIT SUMMARY")
    print("=" * 78)

    print(
        audit[
            [
                "YEAR",
                "STATUS",
                "POSITIVE_TARGETS",
                "INDIVIDUAL_FIRES",
            ]
        ].to_string(
            index=False
        )
    )

    passed_years = int(
        (
            audit["STATUS"]
            == "PASS"
        ).sum()
    )

    failed_years = int(
        (
            audit["STATUS"]
            != "PASS"
        ).sum()
    )

    total_rows = int(
        audit[
            "ROWS"
        ].fillna(0).sum()
    )

    total_positive_targets = int(
        audit[
            "POSITIVE_TARGETS"
        ].fillna(0).sum()
    )

    total_fires = int(
        audit[
            "INDIVIDUAL_FIRES"
        ].fillna(0).sum()
    )

    print()
    print("Years expected:", len(list(YEARS)))
    print("Years passed:", passed_years)
    print("Years failed:", failed_years)

    print(
        "Total feature rows:",
        f"{total_rows:,}",
    )

    print(
        "Positive grid-cell targets:",
        f"{total_positive_targets:,}",
    )

    print(
        "Individual fires represented:",
        f"{total_fires:,}",
    )

    print(
        "\nAudit table saved:",
        audit_path,
    )

    if failed_years:
        raise RuntimeError(
            "One or more feature tables failed validation."
        )

    print(
        "\nALL 2005–2025 FEATURE TABLES PASSED VALIDATION."
    )


if __name__ == "__main__":
    main()
