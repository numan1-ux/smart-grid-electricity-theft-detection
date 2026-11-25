#!/usr/bin/env python3
"""Explainable smart-grid electricity theft screening using consumption patterns.

This educational prototype supports:
  1. A synthetic demonstration: python theft_detection_iot_simple.py --demo
  2. A labelled SGCC-style CSV: python theft_detection_iot_simple.py --data path/to/sgcc.csv

Synthetic results are for demonstrating the program only. A model alert is not
proof of electricity theft and should be reviewed by authorised utility staff.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs_simple")
LABEL_CANDIDATES = ("flag", "label", "target", "class", "theft", "is_theft", "fraud")
ID_CANDIDATES = ("cons_no", "customer_id", "meter_id", "consumer_id", "id")


def generate_demo_data(n_customers: int = 500, n_days: int = 60) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create artificial daily readings to demonstrate the complete workflow."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_theft = int(round(n_customers * 0.15))
    labels = np.zeros(n_customers, dtype=int)
    theft_indices = rng.choice(n_customers, size=n_theft, replace=False)
    labels[theft_indices] = 1

    readings = np.zeros((n_customers, n_days), dtype=float)
    day_factor = 1.0 + 0.08 * np.sin(np.linspace(0, 4 * np.pi, n_days))

    for i in range(n_customers):
        base = float(rng.normal(10.0, 2.2))
        base = max(3.5, base)
        noise = rng.normal(0.0, 1.0, n_days)
        values = base * day_factor + noise
        values = np.clip(values, 0.2, None)

        if labels[i] == 1:
            # Simulate several kinds of suspicious reductions rather than one pattern.
            change_day = int(rng.integers(n_days // 3, n_days // 2 + 1))
            reduction = float(rng.uniform(0.35, 0.75))
            values[change_day:] *= reduction
            if rng.random() < 0.25:
                zero_days = rng.choice(np.arange(change_day, n_days), size=4, replace=False)
                values[zero_days] = rng.uniform(0.0, 0.03, size=len(zero_days))
        else:
            # Some legitimate customers also change usage, creating realistic ambiguity.
            if rng.random() < 0.10:
                change_day = int(rng.integers(n_days // 3, n_days // 2 + 1))
                values[change_day:] *= float(rng.uniform(0.65, 0.90))

        readings[i] = np.clip(values, 0.0, None)

    frame = pd.DataFrame(readings, columns=[f"day_{i + 1:03d}" for i in range(n_days)])
    ids = pd.Series([f"DEMO_{i + 1:04d}" for i in range(n_customers)], name="customer_id")
    return frame, pd.Series(labels, name="label"), ids


def normalise_labels(series: pd.Series) -> pd.Series:
    """Convert common normal/theft labels into 0/1 values."""
    positive = {"1", "theft", "fraud", "fraudulent", "abnormal", "yes", "true", "stolen"}
    negative = {"0", "normal", "genuine", "non-theft", "non theft", "no", "false", "honest"}
    values = series.astype(str).str.strip().str.lower()
    mapped = values.map(lambda value: 1 if value in positive else (0 if value in negative else np.nan))
    if mapped.isna().any():
        examples = sorted(values[mapped.isna()].unique())[:10]
        raise ValueError(
            "The label column must use 0/1 or common normal/theft words. "
            f"Unrecognised label examples: {examples}"
        )
    return mapped.astype(int)


def load_wide_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load a wide CSV with one customer per row and readings in columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        raise ValueError("The input CSV is empty.")

    lookup = {str(column).strip().lower(): column for column in df.columns}
    label_col = next((lookup[name] for name in LABEL_CANDIDATES if name in lookup), None)
    if label_col is None:
        raise ValueError("No label column found. Expected a column such as FLAG, label or target.")
    labels = normalise_labels(df[label_col])

    id_col = next((lookup[name] for name in ID_CANDIDATES if name in lookup), None)
    if id_col is None:
        ids = pd.Series([f"CUSTOMER_{i + 1:06d}" for i in range(len(df))], name="customer_id")
    else:
        ids = df[id_col].astype(str).rename("customer_id")

    excluded = {label_col}
    if id_col is not None:
        excluded.add(id_col)
    candidate_columns = [column for column in df.columns if column not in excluded]
    numeric = df[candidate_columns].apply(pd.to_numeric, errors="coerce")
    reading_columns = [column for column in numeric.columns if numeric[column].notna().any()]
    if len(reading_columns) < 5:
        raise ValueError("At least five numeric consumption-reading columns are required.")
    readings = numeric[reading_columns].replace([np.inf, -np.inf], np.nan)
    return readings, labels, ids


def make_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Calculate six understandable consumption features for each customer."""
    data = readings.astype(float).copy()
    data = data.mask(data < 0)
    row_median = data.median(axis=1)
    data = data.T.fillna(row_median).T.fillna(0.0)
    values = data.to_numpy(dtype=float)

    n_readings = values.shape[1]
    window = max(1, min(15, n_readings // 4))
    average = values.mean(axis=1)
    variation = values.std(axis=1)
    early = values[:, :window].mean(axis=1)
    recent = values[:, -window:].mean(axis=1)
    change = 100.0 * (recent - early) / (early + 1e-8)
    near_zero = (values <= 0.05).mean(axis=1) * 100.0

    return pd.DataFrame(
        {
            "average_daily_kwh": average,
            "usage_variation": variation,
            "near_zero_percent": near_zero,
            "early_average_kwh": early,
            "recent_average_kwh": recent,
            "recent_change_percent": change,
        },
        index=readings.index,
    ).replace([np.inf, -np.inf], np.nan)


def build_model() -> RandomForestClassifier:
    """Create the Random Forest model used in this project."""
    return RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def explain_record(row: pd.Series) -> str:
    """Create a short rule-based explanation for an individual alert."""
    reasons: list[str] = []
    if row["recent_change_percent"] < -35:
        reasons.append("recent use is much lower than earlier use")
    if row["near_zero_percent"] > 10:
        reasons.append("many readings are close to zero")
    if row["usage_variation"] > max(2.5, row["average_daily_kwh"] * 0.45):
        reasons.append("consumption varies unusually around the average")
    if not reasons:
        reasons.append("overall pattern differs from learned normal patterns")
    return "; ".join(reasons[:2])


def save_feature_importance(model: RandomForestClassifier, feature_names: Iterable[str]) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": list(feature_names), "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plot_data = importance.sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 5.5))
    plt.barh(plot_data["feature"], plot_data["importance"])
    plt.xlabel("Feature importance")
    plt.title("Features that influenced the Random Forest model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
    plt.close()
    return importance


def main() -> None:
    parser = argparse.ArgumentParser(description="Explainable electricity-theft screening using consumption patterns")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="Run with artificial demonstration data")
    group.add_argument("--data", type=Path, help="Path to a labelled SGCC-style CSV file")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    if args.demo:
        readings, labels, customer_ids = generate_demo_data()
        mode = "synthetic demonstration"
    else:
        readings, labels, customer_ids = load_wide_dataset(args.data)
        mode = f"dataset: {args.data}"

    features = make_features(readings)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        features,
        labels,
        customer_ids,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }
    pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]).to_csv(
        OUTPUT_DIR / "evaluation_metrics.csv", index=False
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["Actual normal", "Actual theft-labelled"],
        columns=["Predicted normal", "Predicted suspicious"],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    explanation_data = X_test.copy()
    explanation_data["actual_label"] = y_test
    explanation_data["predicted_label"] = predictions
    explanation_data["theft_probability_score"] = probabilities
    explanation_data["explanation"] = explanation_data.apply(explain_record, axis=1)
    explanation_data.insert(0, "customer_id", id_test.values)
    explanation_data.to_csv(OUTPUT_DIR / "customer_predictions.csv", index=False)

    importance = save_feature_importance(model, X_train.columns)

    print(f"Mode: {mode}")
    print("Evaluation results (test split):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f} ({value * 100:.1f}%)")
    print("Confusion matrix (rows = actual [normal, theft]; columns = predicted [normal, suspicious]):")
    print(matrix)
    print("Top features:")
    print(importance.head(6).to_string(index=False))
    print(f"\nFiles saved in: {OUTPUT_DIR.resolve()}")
    print("Reminder: model alerts are screening signals, not proof of theft.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
