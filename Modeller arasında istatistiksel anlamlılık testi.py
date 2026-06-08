import os
import pandas as pd
import numpy as np

from scipy.stats import ttest_rel, wilcoxon, shapiro

project_path = r"C:\Users\zuhir\Desktop\New folder"
os.chdir(project_path)

baseline_file = "improved_baseline_predictions.csv"
fusion_file = "fusion_end_to_end_predictions.csv"

baseline = pd.read_csv(baseline_file)
fusion = pd.read_csv(fusion_file)

print("Baseline shape:", baseline.shape)
print("Fusion shape:", fusion.shape)

print("\nBaseline columns:")
print(baseline.columns.tolist())

print("\nFusion columns:")
print(fusion.columns.tolist())

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
        print("Numeric columns:", numeric_cols)

        if len(numeric_cols) >= 2:
            true_col = numeric_cols[-2]
            pred_col = numeric_cols[-1]

    return true_col, pred_col

b_true_col, b_pred_col = find_true_pred_columns(baseline)
f_true_col, f_pred_col = find_true_pred_columns(fusion)

print("\nBaseline true/pred:", b_true_col, b_pred_col)
print("Fusion true/pred:", f_true_col, f_pred_col)

n = min(len(baseline), len(fusion))

baseline = baseline.iloc[:n].copy()
fusion = fusion.iloc[:n].copy()

baseline_error = np.abs(baseline[b_true_col].values - baseline[b_pred_col].values)
fusion_error = np.abs(fusion[f_true_col].values - fusion[f_pred_col].values)

diff = baseline_error - fusion_error

print("\nNumber of paired samples:", n)
print("Mean baseline absolute error:", baseline_error.mean())
print("Mean fusion absolute error:", fusion_error.mean())
print("Mean error difference baseline - fusion:", diff.mean())

shapiro_stat, shapiro_p = shapiro(diff)

print("\nShapiro normality test:")
print("Statistic:", shapiro_stat)
print("p-value:", shapiro_p)

t_stat, t_p = ttest_rel(baseline_error, fusion_error)

print("\nPaired t-test:")
print("Statistic:", t_stat)
print("p-value:", t_p)

try:
    w_stat, w_p = wilcoxon(baseline_error, fusion_error)
    print("\nWilcoxon signed-rank test:")
    print("Statistic:", w_stat)
    print("p-value:", w_p)
except Exception as e:
    print("\nWilcoxon could not be calculated:")
    print(e)

summary = pd.DataFrame([{
    "n_samples": n,
    "baseline_mean_abs_error": baseline_error.mean(),
    "fusion_mean_abs_error": fusion_error.mean(),
    "mean_error_difference_baseline_minus_fusion": diff.mean(),
    "shapiro_p": shapiro_p,
    "paired_ttest_p": t_p,
    "wilcoxon_p": w_p if "w_p" in locals() else np.nan
}])

summary.to_csv("statistical_test_summary.csv", index=False)

print("\nSaved: statistical_test_summary.csv")