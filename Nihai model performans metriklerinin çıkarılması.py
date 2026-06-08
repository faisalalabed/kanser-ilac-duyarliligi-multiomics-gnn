import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

project_path = r"C:\Users\zuhir\Desktop\New folder"
os.chdir(project_path)

prediction_files = [
    "baseline_predictions.csv",
    "improved_baseline_predictions.csv",
    "fusion_end_to_end_predictions.csv",
    "fusion_with_morgan_predictions.csv"
]

def find_true_pred_columns(df):
    true_candidates = [
        "y_true", "true", "actual", "Actual", "Real", "real",
        "LN_IC50", "true_LN_IC50", "Reference LN_IC50"
    ]

    pred_candidates = [
        "y_pred", "pred", "prediction", "Prediction", "predicted",
        "Predicted", "predicted_LN_IC50", "Predicted LN_IC50"
    ]

    true_col = None
    pred_col = None

    for c in df.columns:
        if c in true_candidates:
            true_col = c
            break

    for c in df.columns:
        if c in pred_candidates:
            pred_col = c
            break

    if true_col is None or pred_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print("Numeric columns found:", numeric_cols)

        if len(numeric_cols) >= 2:
            true_col = numeric_cols[-2]
            pred_col = numeric_cols[-1]

    return true_col, pred_col

results = []

for file in prediction_files:
    if not os.path.exists(file):
        print(f"\nFile not found: {file}")
        continue

    print("\n" + "=" * 80)
    print("Reading:", file)

    df = pd.read_csv(file)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    true_col, pred_col = find_true_pred_columns(df)

    if true_col is None or pred_col is None:
        print("Could not detect true/pred columns for:", file)
        continue

    print("True column:", true_col)
    print("Pred column:", pred_col)

    temp = df[[true_col, pred_col]].dropna().copy()

    y_true = temp[true_col].values
    y_pred = temp[pred_col].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1:
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    else:
        pearson_corr, pearson_p = np.nan, np.nan

    results.append({
        "file": file,
        "n_samples": len(temp),
        "true_column": true_col,
        "pred_column": pred_col,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson_corr,
        "Pearson_p": pearson_p
    })

results_df = pd.DataFrame(results)

print("\n\n===== FINAL METRICS SUMMARY =====")
print(results_df)

results_df.to_csv("final_metrics_summary.csv", index=False)

print("\nSaved: final_metrics_summary.csv")