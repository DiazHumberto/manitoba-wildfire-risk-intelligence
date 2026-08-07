"""Audit all annual Manitoba wildfire-weather outputs from 2005–2025."""

from pathlib import Path

import pandas as pd


YEARS = range(2005, 2026)

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


def find_project_root() -> Path:
    """Find repository root."""

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

    interim_folder = (
        project_root
        / "data"
        / "interim"
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
    # Grid reference
    # ----------------------------------------------------------

    match_path = (
        interim_folder
        / "manitoba_grid_to_era5_land_matches.parquet"
    )

    grid_reference = pd.read_parquet(
        match_path,
        columns=["GRID_ID"],
    )

    grid_count = len(grid_reference)

    if grid_reference["GRID_ID"].duplicated().any():
        raise RuntimeError(
            "ERA5 grid matching table contains duplicate GRID_ID values."
        )

    # ----------------------------------------------------------
    # Original positive fire targets
    # ----------------------------------------------------------

    targets_path = (
        interim_folder
        / "manitoba_daily_fire_targets_2005_2025.parquet"
    )

    fire_targets = pd.read_parquet(
        targets_path,
        columns=[
            "FIRE_DATE",
            "GRID_ID",
            "FIRE_OCCURRED",
            "FIRE_COUNT",
        ],
    )

    fire_targets["FIRE_DATE"] = (
        pd.to_datetime(
            fire_targets["FIRE_DATE"]
        )
        .dt.normalize()
    )

    print("=" * 78)
    print("MANITOBA WILDFIRE RISK INTELLIGENCE")
    print("MULTI-YEAR OUTPUT AUDIT: 2005–2025")
    print("=" * 78)
    print("Grid cells:", grid_count)

    audit_rows = []

    # ----------------------------------------------------------
    # Audit each year independently
    # ----------------------------------------------------------

    for year in YEARS:

        print("\n" + "-" * 78)
        print("AUDITING YEAR:", year)

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

        weather_path = (
            interim_folder
            / (
                "era5_grid_weather_"
                f"{year}_march_october.parquet"
            )
        )

        model_path = (
            interim_folder
            / (
                "manitoba_model_table_"
                f"{year}_fire_season.parquet"
            )
        )

        weather_exists = weather_path.exists()
        model_exists = model_path.exists()

        if not weather_exists or not model_exists:

            audit_rows.append(
                {
                    "YEAR": year,
                    "STATUS": "FAIL",
                    "WEATHER_EXISTS": weather_exists,
                    "MODEL_EXISTS": model_exists,
                }
            )

            print("FAIL — annual file missing.")
            continue

        # ------------------------------------------------------
        # Weather table
        # ------------------------------------------------------

        weather = pd.read_parquet(
            weather_path,
            columns=[
                "LOCAL_DATE",
                "GRID_ID",
                *WEATHER_VARIABLES,
            ],
        )

        weather["LOCAL_DATE"] = (
            pd.to_datetime(
                weather["LOCAL_DATE"]
            )
            .dt.normalize()
        )

        weather_rows = len(weather)

        weather_dates = (
            weather["LOCAL_DATE"]
            .nunique()
        )

        weather_grid_cells = (
            weather["GRID_ID"]
            .nunique()
        )

        weather_duplicates = int(
            weather.duplicated(
                subset=[
                    "LOCAL_DATE",
                    "GRID_ID",
                ]
            ).sum()
        )

        weather_missing = int(
            weather[
                WEATHER_VARIABLES
            ]
            .isna()
            .sum()
            .sum()
        )

        weather_first_date = (
            weather["LOCAL_DATE"].min()
        )

        weather_last_date = (
            weather["LOCAL_DATE"].max()
        )

        # ------------------------------------------------------
        # Model table
        # ------------------------------------------------------

        model = pd.read_parquet(
            model_path,
            columns=[
                "LOCAL_DATE",
                "GRID_ID",
                *WEATHER_VARIABLES,
                "FIRE_OCCURRED",
                "FIRE_COUNT",
            ],
        )

        model["LOCAL_DATE"] = (
            pd.to_datetime(
                model["LOCAL_DATE"]
            )
            .dt.normalize()
        )

        model_rows = len(model)

        model_dates = (
            model["LOCAL_DATE"]
            .nunique()
        )

        model_grid_cells = (
            model["GRID_ID"]
            .nunique()
        )

        model_duplicates = int(
            model.duplicated(
                subset=[
                    "LOCAL_DATE",
                    "GRID_ID",
                ]
            ).sum()
        )

        model_missing_weather = int(
            model[
                WEATHER_VARIABLES
            ]
            .isna()
            .sum()
            .sum()
        )

        positive_grid_cell_days = int(
            model[
                "FIRE_OCCURRED"
            ].sum()
        )

        individual_fires = int(
            model[
                "FIRE_COUNT"
            ].sum()
        )

        model_first_date = (
            model["LOCAL_DATE"].min()
        )

        model_last_date = (
            model["LOCAL_DATE"].max()
        )

        # ------------------------------------------------------
        # Compare with original target table
        # ------------------------------------------------------

        year_targets = fire_targets.loc[
            fire_targets[
                "FIRE_DATE"
            ].between(
                f"{year}-04-01",
                f"{year}-10-31",
            )
        ].copy()

        expected_positive_days = len(
            year_targets
        )

        expected_individual_fires = int(
            year_targets[
                "FIRE_COUNT"
            ].sum()
        )

        # ------------------------------------------------------
        # Validation checks
        # ------------------------------------------------------

        checks = {
            "weather_rows":
                weather_rows
                == expected_weather_rows,

            "model_rows":
                model_rows
                == expected_model_rows,

            "weather_dates":
                weather_dates
                == len(support_dates),

            "model_dates":
                model_dates
                == len(fire_season_dates),

            "weather_grid_cells":
                weather_grid_cells
                == grid_count,

            "model_grid_cells":
                model_grid_cells
                == grid_count,

            "weather_duplicates":
                weather_duplicates == 0,

            "model_duplicates":
                model_duplicates == 0,

            "weather_missing":
                weather_missing == 0,

            "model_missing_weather":
                model_missing_weather == 0,

            "positive_targets":
                positive_grid_cell_days
                == expected_positive_days,

            "individual_fires":
                individual_fires
                == expected_individual_fires,

            "weather_start":
                weather_first_date
                == pd.Timestamp(
                    f"{year}-03-01"
                ),

            "weather_end":
                weather_last_date
                == pd.Timestamp(
                    f"{year}-10-31"
                ),

            "model_start":
                model_first_date
                == pd.Timestamp(
                    f"{year}-04-01"
                ),

            "model_end":
                model_last_date
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

        status = (
            "PASS"
            if not failed_checks
            else "FAIL"
        )

        weather_size_mb = (
            weather_path.stat().st_size
            / 1_000_000
        )

        model_size_mb = (
            model_path.stat().st_size
            / 1_000_000
        )

        audit_rows.append(
            {
                "YEAR": year,
                "STATUS": status,

                "WEATHER_ROWS":
                    weather_rows,

                "EXPECTED_WEATHER_ROWS":
                    expected_weather_rows,

                "MODEL_ROWS":
                    model_rows,

                "EXPECTED_MODEL_ROWS":
                    expected_model_rows,

                "WEATHER_DATES":
                    weather_dates,

                "MODEL_DATES":
                    model_dates,

                "GRID_CELLS":
                    model_grid_cells,

                "WEATHER_MISSING":
                    weather_missing,

                "MODEL_WEATHER_MISSING":
                    model_missing_weather,

                "WEATHER_DUPLICATES":
                    weather_duplicates,

                "MODEL_DUPLICATES":
                    model_duplicates,

                "POSITIVE_GRID_CELL_DAYS":
                    positive_grid_cell_days,

                "EXPECTED_POSITIVE_DAYS":
                    expected_positive_days,

                "INDIVIDUAL_FIRES":
                    individual_fires,

                "EXPECTED_INDIVIDUAL_FIRES":
                    expected_individual_fires,

                "WEATHER_SIZE_MB":
                    round(
                        weather_size_mb,
                        2,
                    ),

                "MODEL_SIZE_MB":
                    round(
                        model_size_mb,
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
            "Weather rows:",
            weather_rows,
        )
        print(
            "Model rows:",
            model_rows,
        )
        print(
            "Missing weather:",
            weather_missing,
        )
        print(
            "Duplicate model rows:",
            model_duplicates,
        )
        print(
            "Positive grid-cell days:",
            positive_grid_cell_days,
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

        # Explicitly release large annual tables
        del weather
        del model

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------

    audit = pd.DataFrame(
        audit_rows
    )

    audit_path = (
        output_folder
        / "multiyear_data_audit_2005_2025.csv"
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    print("\n" + "=" * 78)
    print("FINAL MULTI-YEAR SUMMARY")
    print("=" * 78)

    print(
        audit[
            [
                "YEAR",
                "STATUS",
                "POSITIVE_GRID_CELL_DAYS",
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

    print("\nYears expected:", len(list(YEARS)))
    print("Years passed:", passed_years)
    print("Years failed:", failed_years)

    if "INDIVIDUAL_FIRES" in audit.columns:
        print(
            "Fire-season fires represented:",
            int(
                audit[
                    "INDIVIDUAL_FIRES"
                ].fillna(0).sum()
            ),
        )

    print(
        "\nAudit table saved:",
        audit_path,
    )

    if failed_years:
        raise RuntimeError(
            "One or more years failed the multi-year audit."
        )

    print(
        "\nALL 2005–2025 ANNUAL OUTPUTS PASSED VALIDATION."
    )


if __name__ == "__main__":
    main()
