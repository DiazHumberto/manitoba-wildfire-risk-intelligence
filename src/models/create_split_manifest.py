"""Create and validate chronological model splits for wildfire prediction."""

from pathlib import Path

import pandas as pd


TRAIN_YEARS = range(2005, 2020)
VALIDATION_YEARS = range(2020, 2023)
TEST_YEARS = range(2023, 2026)

EXPECTED_GRID_CELLS = 6501
EXPECTED_FORECAST_DAYS = 214
EXPECTED_ROWS_PER_YEAR = (
    EXPECTED_GRID_CELLS
    * EXPECTED_FORECAST_DAYS
)


def find_project_root() -> Path:
    """Locate repository root."""

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


def assign_split(year: int) -> str:
    """Assign a year to train, validation, or test."""

    if year in TRAIN_YEARS:
        return "train"

    if year in VALIDATION_YEARS:
        return "validation"

    if year in TEST_YEARS:
        return "test"

    raise ValueError(
        f"Year {year} is outside the modelling period."
    )


def main() -> None:

    project_root = find_project_root()

    feature_folder = (
        project_root
        / "data"
        / "processed"
        / "features"
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

    all_years = list(
        range(2005, 2026)
    )

    print("=" * 76)
    print("MANITOBA WILDFIRE RISK INTELLIGENCE")
    print("CHRONOLOGICAL MODEL SPLIT MANIFEST")
    print("=" * 76)

    print()
    print("TRAIN:      2005–2019")
    print("VALIDATION: 2020–2022")
    print("TEST:       2023–2025")

    manifest_rows = []

    # ----------------------------------------------------------
    # Inspect each annual feature table
    # ----------------------------------------------------------

    for year in all_years:

        split = assign_split(year)

        feature_path = (
            feature_folder
            / f"manitoba_next_day_features_{year}.parquet"
        )

        print()
        print("-" * 76)
        print(
            f"{year}  →  {split.upper()}"
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing feature table: {feature_path}"
            )

        # Only read columns needed for split statistics.
        data = pd.read_parquet(
            feature_path,
            columns=[
                "PREDICTOR_DATE",
                "FORECAST_DATE",
                "YEAR",
                "GRID_ID",
                "FIRE_OCCURRED_NEXT_DAY",
                "FIRE_COUNT_NEXT_DAY",
            ],
        )

        data[
            "FORECAST_DATE"
        ] = pd.to_datetime(
            data["FORECAST_DATE"]
        ).dt.normalize()

        rows = len(data)

        forecast_days = (
            data[
                "FORECAST_DATE"
            ].nunique()
        )

        grid_cells = (
            data[
                "GRID_ID"
            ].nunique()
        )

        positive_targets = int(
            data[
                "FIRE_OCCURRED_NEXT_DAY"
            ].sum()
        )

        individual_fires = int(
            data[
                "FIRE_COUNT_NEXT_DAY"
            ].sum()
        )

        prevalence = (
            positive_targets
            / rows
        )

        duplicate_rows = int(
            data.duplicated(
                subset=[
                    "FORECAST_DATE",
                    "GRID_ID",
                ]
            ).sum()
        )

        invalid_year_rows = int(
            (
                data["YEAR"]
                != year
            ).sum()
        )

        first_date = (
            data[
                "FORECAST_DATE"
            ].min()
        )

        last_date = (
            data[
                "FORECAST_DATE"
            ].max()
        )

        expected_first_date = pd.Timestamp(
            f"{year}-04-01"
        )

        expected_last_date = pd.Timestamp(
            f"{year}-10-31"
        )

        # ------------------------------------------------------
        # Validate annual table
        # ------------------------------------------------------

        checks = {
            "rows":
                rows
                == EXPECTED_ROWS_PER_YEAR,

            "forecast_days":
                forecast_days
                == EXPECTED_FORECAST_DAYS,

            "grid_cells":
                grid_cells
                == EXPECTED_GRID_CELLS,

            "duplicates":
                duplicate_rows == 0,

            "year":
                invalid_year_rows == 0,

            "start_date":
                first_date
                == expected_first_date,

            "end_date":
                last_date
                == expected_last_date,
        }

        failed_checks = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        if failed_checks:
            raise RuntimeError(
                f"{year} failed checks: "
                + ", ".join(
                    failed_checks
                )
            )

        print(
            "Rows:",
            f"{rows:,}",
        )

        print(
            "Positive targets:",
            f"{positive_targets:,}",
        )

        print(
            "Individual fires:",
            f"{individual_fires:,}",
        )

        print(
            "Positive prevalence:",
            f"{prevalence:.6%}",
        )

        manifest_rows.append(
            {
                "YEAR":
                    year,

                "SPLIT":
                    split,

                "FILE":
                    str(
                        feature_path.relative_to(
                            project_root
                        )
                    ),

                "ROWS":
                    rows,

                "FORECAST_DAYS":
                    forecast_days,

                "GRID_CELLS":
                    grid_cells,

                "POSITIVE_TARGETS":
                    positive_targets,

                "INDIVIDUAL_FIRES":
                    individual_fires,

                "POSITIVE_PREVALENCE":
                    prevalence,

                "FIRST_FORECAST_DATE":
                    first_date.date(),

                "LAST_FORECAST_DATE":
                    last_date.date(),

                "FILE_SIZE_MB":
                    round(
                        feature_path.stat().st_size
                        / 1_000_000,
                        2,
                    ),
            }
        )

        del data

    # ----------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        output_folder
        / "model_split_manifest_2005_2025.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    # ----------------------------------------------------------
    # Split summary
    # ----------------------------------------------------------

    split_order = [
        "train",
        "validation",
        "test",
    ]

    summary_rows = []

    for split in split_order:

        subset = manifest.loc[
            manifest["SPLIT"]
            == split
        ]

        rows = int(
            subset["ROWS"].sum()
        )

        positive_targets = int(
            subset[
                "POSITIVE_TARGETS"
            ].sum()
        )

        individual_fires = int(
            subset[
                "INDIVIDUAL_FIRES"
            ].sum()
        )

        prevalence = (
            positive_targets
            / rows
        )

        summary_rows.append(
            {
                "SPLIT":
                    split,

                "START_YEAR":
                    int(
                        subset["YEAR"].min()
                    ),

                "END_YEAR":
                    int(
                        subset["YEAR"].max()
                    ),

                "YEARS":
                    len(subset),

                "ROWS":
                    rows,

                "POSITIVE_TARGETS":
                    positive_targets,

                "NEGATIVE_TARGETS":
                    rows
                    - positive_targets,

                "INDIVIDUAL_FIRES":
                    individual_fires,

                "POSITIVE_PREVALENCE":
                    prevalence,

                "POSITIVE_PERCENT":
                    prevalence * 100,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        output_folder
        / "model_split_summary_2005_2025.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ----------------------------------------------------------
    # Global checks
    # ----------------------------------------------------------

    total_rows = int(
        manifest[
            "ROWS"
        ].sum()
    )

    total_positive = int(
        manifest[
            "POSITIVE_TARGETS"
        ].sum()
    )

    total_fires = int(
        manifest[
            "INDIVIDUAL_FIRES"
        ].sum()
    )

    if len(manifest) != 21:
        raise RuntimeError(
            "Expected exactly 21 annual files."
        )

    if total_rows != 29_215_494:
        raise RuntimeError(
            "Unexpected total row count: "
            f"{total_rows:,}"
        )

    if total_positive != 7_324:
        raise RuntimeError(
            "Unexpected positive target count: "
            f"{total_positive:,}"
        )

    if total_fires != 7_882:
        raise RuntimeError(
            "Unexpected individual fire count: "
            f"{total_fires:,}"
        )

    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------

    print()
    print("=" * 76)
    print("MODEL SPLIT SUMMARY")
    print("=" * 76)

    for row in summary.itertuples():

        print()
        print(
            f"{row.SPLIT.upper()} "
            f"({row.START_YEAR}–{row.END_YEAR})"
        )

        print(
            "  Years:",
            row.YEARS,
        )

        print(
            "  Rows:",
            f"{row.ROWS:,}",
        )

        print(
            "  Positive targets:",
            f"{row.POSITIVE_TARGETS:,}",
        )

        print(
            "  Negative targets:",
            f"{row.NEGATIVE_TARGETS:,}",
        )

        print(
            "  Individual fires:",
            f"{row.INDIVIDUAL_FIRES:,}",
        )

        print(
            "  Positive prevalence:",
            f"{row.POSITIVE_PERCENT:.4f}%",
        )

    print()
    print("=" * 76)
    print("FULL DATASET")
    print("=" * 76)

    print(
        "Rows:",
        f"{total_rows:,}",
    )

    print(
        "Positive targets:",
        f"{total_positive:,}",
    )

    print(
        "Individual fires:",
        f"{total_fires:,}",
    )

    print()
    print(
        "Manifest saved:",
        manifest_path,
    )

    print(
        "Split summary saved:",
        summary_path,
    )

    print()
    print(
        "CHRONOLOGICAL MODEL SPLITS PASSED VALIDATION."
    )


if __name__ == "__main__":
    main()
