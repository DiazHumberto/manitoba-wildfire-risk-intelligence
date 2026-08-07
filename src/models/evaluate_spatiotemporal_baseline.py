"""Evaluate a historical spatial + seasonal wildfire-risk baseline.

Training:
    2005–2019

Validation:
    2020–2022

Test:
    2023–2025 NOT USED

The score is learned from historical wildfire occurrence for each
GRID_ID × calendar month combination.

A small empirical-Bayes prior shrinks sparse cell-month estimates
toward the overall historical rate for that month.
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

# Equivalent to approximately one month of prior observations.
# This prevents cell-month combinations with very few historical
# fires from receiving unstable probability estimates.
PRIOR_OBSERVATIONS = 30.0


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


def load_target_data(
    feature_folder: Path,
    years,
) -> pd.DataFrame:
    """Load only columns needed for this baseline."""

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

        frame[
            "FORECAST_DATE"
        ] = pd.to_datetime(
            frame["FORECAST_DATE"]
        )

        frame[
            "MONTH"
        ] = (
            frame[
                "FORECAST_DATE"
            ]
            .dt.month
            .astype("int8")
        )

        frames.append(
            frame[
                [
                    "GRID_ID",
                    "MONTH",
                    "FIRE_OCCURRED_NEXT_DAY",
                ]
            ]
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def calculate_top_risk_metrics(
    data: pd.DataFrame,
    fractions=(0.01, 0.05, 0.10),
) -> pd.DataFrame:
    """Measure fire capture within highest-risk observations."""

    ranked = data.sort_values(
        "SPATIOTEMPORAL_RISK_SCORE",
        ascending=False,
    ).reset_index(drop=True)

    total_rows = len(
        ranked
    )

    total_positive = int(
        ranked[
            "FIRE_OCCURRED_NEXT_DAY"
        ].sum()
    )

    prevalence = (
        total_positive
        / total_rows
    )

    results = []

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

        selected = ranked.iloc[
            :n_selected
        ]

        captured = int(
            selected[
                "FIRE_OCCURRED_NEXT_DAY"
            ].sum()
        )

        capture_rate = (
            captured
            / total_positive
        )

        precision = (
            captured
            / n_selected
        )

        precision_lift = (
            precision
            / prevalence
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
                    captured,

                "POSITIVE_CAPTURE_RATE":
                    capture_rate,

                "PRECISION":
                    precision,

                "PRECISION_LIFT":
                    precision_lift,
            }
        )

    return pd.DataFrame(
        results
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

    print("=" * 78)
    print("MANITOBA WILDFIRE RISK INTELLIGENCE")
    print("SPATIAL + SEASONAL HISTORICAL BASELINE")
    print("=" * 78)

    print()
    print("Training years:   2005–2019")
    print("Validation years: 2020–2022")
    print("Test years:       NOT USED")
    print(
        "Prior observations:",
        PRIOR_OBSERVATIONS,
    )

    # ==========================================================
    # TRAINING DATA
    # ==========================================================

    print()
    print("=" * 78)
    print("TRAINING")
    print("=" * 78)
    print()

    train = load_target_data(
        feature_folder,
        TRAIN_YEARS,
    )

    train_rows = len(
        train
    )

    train_positive = int(
        train[
            "FIRE_OCCURRED_NEXT_DAY"
        ].sum()
    )

    train_prevalence = (
        train_positive
        / train_rows
    )

    print()
    print(
        "Training rows:",
        f"{train_rows:,}",
    )

    print(
        "Training positives:",
        f"{train_positive:,}",
    )

    print(
        "Training prevalence:",
        f"{train_prevalence:.6%}",
    )

    # ==========================================================
    # MONTH-LEVEL HISTORICAL RISK
    # ==========================================================

    month_risk = (
        train
        .groupby(
            "MONTH",
            as_index=False,
        )
        .agg(
            MONTH_OBSERVATIONS=(
                "FIRE_OCCURRED_NEXT_DAY",
                "size",
            ),
            MONTH_POSITIVES=(
                "FIRE_OCCURRED_NEXT_DAY",
                "sum",
            ),
        )
    )

    month_risk[
        "MONTH_RISK"
    ] = (
        month_risk[
            "MONTH_POSITIVES"
        ]
        / month_risk[
            "MONTH_OBSERVATIONS"
        ]
    )

    print()
    print("Historical monthly fire rates:")
    print()

    for row in month_risk.itertuples():

        print(
            f"Month {row.MONTH}: "
            f"{row.MONTH_POSITIVES:,} positives — "
            f"{row.MONTH_RISK:.5%}"
        )

    # ==========================================================
    # GRID × MONTH HISTORICAL RISK
    # ==========================================================

    cell_month = (
        train
        .groupby(
            [
                "GRID_ID",
                "MONTH",
            ],
            as_index=False,
        )
        .agg(
            CELL_MONTH_OBSERVATIONS=(
                "FIRE_OCCURRED_NEXT_DAY",
                "size",
            ),
            CELL_MONTH_POSITIVES=(
                "FIRE_OCCURRED_NEXT_DAY",
                "sum",
            ),
        )
    )

    cell_month = cell_month.merge(
        month_risk[
            [
                "MONTH",
                "MONTH_RISK",
            ]
        ],
        on="MONTH",
        how="left",
        validate="many_to_one",
    )

    # ----------------------------------------------------------
    # Empirical-Bayes smoothing
    #
    # score =
    #
    # observed positive events
    # + prior strength × historical monthly prevalence
    #
    # divided by
    #
    # observations + prior strength
    # ----------------------------------------------------------

    cell_month[
        "SPATIOTEMPORAL_RISK_SCORE"
    ] = (
        (
            cell_month[
                "CELL_MONTH_POSITIVES"
            ]
            +
            PRIOR_OBSERVATIONS
            * cell_month[
                "MONTH_RISK"
            ]
        )
        /
        (
            cell_month[
                "CELL_MONTH_OBSERVATIONS"
            ]
            + PRIOR_OBSERVATIONS
        )
    )

    expected_groups = (
        6501
        * 7
    )

    if len(cell_month) != expected_groups:

        raise RuntimeError(
            "Unexpected number of grid-month groups: "
            f"{len(cell_month):,} != {expected_groups:,}"
        )

    print()
    print(
        "Grid-month risk groups:",
        f"{len(cell_month):,}",
    )

    print(
        "Groups with ≥1 historical fire:",
        f"{int((cell_month['CELL_MONTH_POSITIVES'] > 0).sum()):,}",
    )

    print(
        "Maximum smoothed risk:",
        f"{cell_month['SPATIOTEMPORAL_RISK_SCORE'].max():.4%}",
    )

    risk_table_path = (
        output_folder
        / "historical_spatiotemporal_risk_train_2005_2019.csv"
    )

    cell_month.to_csv(
        risk_table_path,
        index=False,
    )

    del train

    # ==========================================================
    # VALIDATION
    # ==========================================================

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
        cell_month[
            [
                "GRID_ID",
                "MONTH",
                "SPATIOTEMPORAL_RISK_SCORE",
            ]
        ],
        on=[
            "GRID_ID",
            "MONTH",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_scores = int(
        validation[
            "SPATIOTEMPORAL_RISK_SCORE"
        ]
        .isna()
        .sum()
    )

    if missing_scores:

        raise RuntimeError(
            f"{missing_scores:,} validation rows "
            "do not have historical risk scores."
        )

    y_true = validation[
        "FIRE_OCCURRED_NEXT_DAY"
    ].to_numpy()

    y_score = validation[
        "SPATIOTEMPORAL_RISK_SCORE"
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

    # ==========================================================
    # METRICS
    # ==========================================================

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

    pr_lift = (
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
    print("SPATIOTEMPORAL BASELINE METRICS")
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
        f"{pr_lift:.2f}x",
    )

    print(
        "ROC-AUC:",
        f"{roc_auc:.6f}",
    )

    print(
        "Brier score:",
        f"{brier:.8f}",
    )

    # ==========================================================
    # TOP-RISK CAPTURE
    # ==========================================================

    top_metrics = (
        calculate_top_risk_metrics(
            validation
        )
    )

    print()
    print("TOP-RISK CAPTURE")
    print("-" * 78)

    for row in top_metrics.itertuples():

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

        print(
            "  Precision lift:",
            f"{row.PRECISION_LIFT:.2f}x",
        )

    # ==========================================================
    # COMPARE WITH SPATIAL-ONLY BASELINE
    # ==========================================================

    spatial_metrics_path = (
        output_folder
        / "baseline_spatial_validation_metrics.csv"
    )

    if spatial_metrics_path.exists():

        spatial_metrics = pd.read_csv(
            spatial_metrics_path
        )

        spatial_ap = float(
            spatial_metrics.iloc[0][
                "AVERAGE_PRECISION"
            ]
        )

        spatial_roc = float(
            spatial_metrics.iloc[0][
                "ROC_AUC"
            ]
        )

        print()
        print("=" * 78)
        print("COMPARISON WITH SPATIAL-ONLY BASELINE")
        print("=" * 78)

        print(
            "Spatial PR-AUC:",
            f"{spatial_ap:.6f}",
        )

        print(
            "Spatial + seasonal PR-AUC:",
            f"{average_precision:.6f}",
        )

        print(
            "Absolute PR-AUC change:",
            f"{average_precision - spatial_ap:+.6f}",
        )

        print(
            "Relative PR-AUC change:",
            f"{((average_precision / spatial_ap) - 1) * 100:+.2f}%",
        )

        print()

        print(
            "Spatial ROC-AUC:",
            f"{spatial_roc:.6f}",
        )

        print(
            "Spatial + seasonal ROC-AUC:",
            f"{roc_auc:.6f}",
        )

    # ==========================================================
    # SAVE RESULTS
    # ==========================================================

    metrics = pd.DataFrame(
        [
            {
                "MODEL":
                    "historical_spatial_seasonal_risk",

                "TRAIN_START_YEAR":
                    2005,

                "TRAIN_END_YEAR":
                    2019,

                "EVAL_START_YEAR":
                    2020,

                "EVAL_END_YEAR":
                    2022,

                "PRIOR_OBSERVATIONS":
                    PRIOR_OBSERVATIONS,

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
                    pr_lift,

                "ROC_AUC":
                    roc_auc,

                "BRIER_SCORE":
                    brier,
            }
        ]
    )

    metrics_path = (
        output_folder
        / "baseline_spatiotemporal_validation_metrics.csv"
    )

    top_metrics_path = (
        output_folder
        / "baseline_spatiotemporal_top_risk_capture.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    top_metrics.to_csv(
        top_metrics_path,
        index=False,
    )

    print()
    print("=" * 78)
    print("SPATIOTEMPORAL BASELINE COMPLETED")
    print("=" * 78)

    print(
        "Risk table:",
        risk_table_path,
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
