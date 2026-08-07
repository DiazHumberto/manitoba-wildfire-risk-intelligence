"""Evaluate a historical spatial wildfire-risk baseline.

Training period:
    2005–2019

Validation period:
    2020–2022

The model assigns each 10-km grid cell a historical wildfire
probability based only on the training years.

The final test period (2023–2025) is intentionally not read here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


TRAIN_YEARS = range(2005, 2020)
VALIDATION_YEARS = range(2020, 2023)


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


def load_target_data(
    feature_folder: Path,
    years,
) -> pd.DataFrame:
    """Load only columns needed for the baseline."""

    frames = []

    for year in years:

        path = (
            feature_folder
            / f"manitoba_next_day_features_{year}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing feature table: {path}"
            )

        print(
            f"Loading {year}:",
            path.name,
        )

        frame = pd.read_parquet(
            path,
            columns=[
                "FORECAST_DATE",
                "GRID_ID",
                "FIRE_OCCURRED_NEXT_DAY",
            ],
        )

        frame["YEAR"] = year

        frames.append(
            frame
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def calculate_top_risk_metrics(
    data: pd.DataFrame,
    fractions=(0.01, 0.05, 0.10),
) -> list[dict]:
    """Calculate fire capture among highest-risk observations."""

    results = []

    data = data.sort_values(
        "SPATIAL_RISK_SCORE",
        ascending=False,
    ).reset_index(drop=True)

    total_positive = int(
        data[
            "FIRE_OCCURRED_NEXT_DAY"
        ].sum()
    )

    total_rows = len(data)

    for fraction in fractions:

        n_selected = max(
            1,
            int(
                np.ceil(
                    total_rows
                    * fraction
                )
            ),
        )

        selected = data.iloc[
            :n_selected
        ]

        captured_positive = int(
            selected[
                "FIRE_OCCURRED_NEXT_DAY"
            ].sum()
        )

        capture_rate = (
            captured_positive
            / total_positive
            if total_positive
            else np.nan
        )

        precision = (
            captured_positive
            / n_selected
        )

        results.append(
            {
                "TOP_FRACTION":
                    fraction,

                "TOP_PERCENT":
                    fraction * 100,

                "ROWS_SELECTED":
                    n_selected,

                "POSITIVES_CAPTURED":
                    captured_positive,

                "POSITIVE_CAPTURE_RATE":
                    capture_rate,

                "PRECISION":
                    precision,
            }
        )

    return results


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

    print("=" * 78)
    print("MANITOBA WILDFIRE RISK INTELLIGENCE")
    print("HISTORICAL SPATIAL-RISK BASELINE")
    print("=" * 78)

    print()
    print("Training years:   2005–2019")
    print("Validation years: 2020–2022")
    print("Test years:       NOT USED")
    print()

    # ----------------------------------------------------------
    # Load training targets
    # ----------------------------------------------------------

    train = load_target_data(
        feature_folder,
        TRAIN_YEARS,
    )

    print()
    print(
        "Training rows:",
        f"{len(train):,}",
    )

    print(
        "Training positives:",
        f"{int(train['FIRE_OCCURRED_NEXT_DAY'].sum()):,}",
    )

    global_train_rate = (
        train[
            "FIRE_OCCURRED_NEXT_DAY"
        ].mean()
    )

    print(
        "Global training prevalence:",
        f"{global_train_rate:.6%}",
    )

    # ----------------------------------------------------------
    # Historical risk by grid cell
    # ----------------------------------------------------------

    grid_risk = (
        train
        .groupby(
            "GRID_ID",
            as_index=False,
        )
        .agg(
            TRAIN_OBSERVATIONS=(
                "FIRE_OCCURRED_NEXT_DAY",
                "size",
            ),
            TRAIN_POSITIVES=(
                "FIRE_OCCURRED_NEXT_DAY",
                "sum",
            ),
        )
    )

    grid_risk[
        "SPATIAL_RISK_SCORE"
    ] = (
        grid_risk[
            "TRAIN_POSITIVES"
        ]
        / grid_risk[
            "TRAIN_OBSERVATIONS"
        ]
    )

    print()
    print(
        "Grid cells learned:",
        f"{len(grid_risk):,}",
    )

    print(
        "Grid cells with ≥1 training fire day:",
        f"{int((grid_risk['TRAIN_POSITIVES'] > 0).sum()):,}",
    )

    print(
        "Maximum historical grid risk:",
        f"{grid_risk['SPATIAL_RISK_SCORE'].max():.4%}",
    )

    grid_risk_path = (
        output_folder
        / "historical_spatial_risk_train_2005_2019.csv"
    )

    grid_risk.to_csv(
        grid_risk_path,
        index=False,
    )

    # Training data no longer needed.
    del train

    # ----------------------------------------------------------
    # Load validation targets
    # ----------------------------------------------------------

    print()
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    print()

    validation = load_target_data(
        feature_folder,
        VALIDATION_YEARS,
    )

    validation = validation.merge(
        grid_risk[
            [
                "GRID_ID",
                "SPATIAL_RISK_SCORE",
            ]
        ],
        on="GRID_ID",
        how="left",
        validate="many_to_one",
    )

    # All 6,501 grid cells should exist in training.
    missing_scores = int(
        validation[
            "SPATIAL_RISK_SCORE"
        ].isna().sum()
    )

    if missing_scores:
        raise RuntimeError(
            f"Validation contains {missing_scores:,} rows "
            "without a spatial risk score."
        )

    y_true = validation[
        "FIRE_OCCURRED_NEXT_DAY"
    ].to_numpy()

    y_score = validation[
        "SPATIAL_RISK_SCORE"
    ].to_numpy()

    validation_rows = len(
        validation
    )

    validation_positive = int(
        y_true.sum()
    )

    validation_prevalence = (
        validation_positive
        / validation_rows
    )

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    average_precision = (
        average_precision_score(
            y_true,
            y_score,
        )
    )

    roc_auc = (
        roc_auc_score(
            y_true,
            y_score,
        )
    )

    brier = (
        brier_score_loss(
            y_true,
            y_score,
        )
    )

    lift_over_random = (
        average_precision
        / validation_prevalence
    )

    print(
        "Validation rows:",
        f"{validation_rows:,}",
    )

    print(
        "Positive targets:",
        f"{validation_positive:,}",
    )

    print(
        "Positive prevalence:",
        f"{validation_prevalence:.6%}",
    )

    print()
    print("BASELINE METRICS")
    print("-" * 78)

    print(
        "Average Precision (PR-AUC):",
        f"{average_precision:.6f}",
    )

    print(
        "Random PR baseline:",
        f"{validation_prevalence:.6f}",
    )

    print(
        "Lift over random:",
        f"{lift_over_random:.2f}x",
    )

    print(
        "ROC-AUC:",
        f"{roc_auc:.6f}",
    )

    print(
        "Brier score:",
        f"{brier:.8f}",
    )

    # ----------------------------------------------------------
    # Top-risk capture
    # ----------------------------------------------------------

    top_metrics = (
        calculate_top_risk_metrics(
            validation,
        )
    )

    top_metrics_df = pd.DataFrame(
        top_metrics
    )

    print()
    print("TOP-RISK CAPTURE")
    print("-" * 78)

    for row in top_metrics_df.itertuples():

        print(
            f"Top {row.TOP_PERCENT:.0f}% of grid-days:"
        )

        print(
            "  Fires captured:",
            f"{row.POSITIVES_CAPTURED:,}",
        )

        print(
            "  Capture rate:",
            f"{row.POSITIVE_CAPTURE_RATE:.2%}",
        )

        print(
            "  Precision:",
            f"{row.PRECISION:.4%}",
        )

    # ----------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------

    metrics = pd.DataFrame(
        [
            {
                "MODEL":
                    "historical_spatial_risk",

                "TRAIN_START_YEAR":
                    2005,

                "TRAIN_END_YEAR":
                    2019,

                "EVAL_START_YEAR":
                    2020,

                "EVAL_END_YEAR":
                    2022,

                "VALIDATION_ROWS":
                    validation_rows,

                "VALIDATION_POSITIVES":
                    validation_positive,

                "PREVALENCE":
                    validation_prevalence,

                "AVERAGE_PRECISION":
                    average_precision,

                "RANDOM_PR_BASELINE":
                    validation_prevalence,

                "PR_LIFT_OVER_RANDOM":
                    lift_over_random,

                "ROC_AUC":
                    roc_auc,

                "BRIER_SCORE":
                    brier,
            }
        ]
    )

    metrics_path = (
        output_folder
        / "baseline_spatial_validation_metrics.csv"
    )

    top_metrics_path = (
        output_folder
        / "baseline_spatial_top_risk_capture.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    top_metrics_df.to_csv(
        top_metrics_path,
        index=False,
    )

    print()
    print("=" * 78)
    print("BASELINE COMPLETED")
    print("=" * 78)

    print(
        "Grid-risk table:",
        grid_risk_path,
    )

    print(
        "Metrics:",
        metrics_path,
    )

    print(
        "Top-risk metrics:",
        top_metrics_path,
    )

    print()
    print(
        "2023–2025 TEST DATA WAS NOT USED."
    )


if __name__ == "__main__":
    main()
