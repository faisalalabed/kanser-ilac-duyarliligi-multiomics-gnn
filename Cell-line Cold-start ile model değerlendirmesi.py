import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr

project_path = r"C:\Users\zuhir\Desktop\New folder"
os.chdir(project_path)

data = pd.read_csv("baseline_merged_dataset.csv")
print("Original data shape:", data.shape)

if "DRUG_NAME_x" in data.columns:
    data = data.rename(columns={"DRUG_NAME_x": "DRUG_NAME"})

if "DRUG_NAME_y" in data.columns:
    data = data.drop(columns=["DRUG_NAME_y"])

print("After cleaning duplicate columns:", data.shape)

target_col = "LN_IC50"

drop_cols = [
    "LN_IC50",
    "CELL_LINE_NAME",
    "model_id",
    "DRUG_NAME"
]

drop_cols = [c for c in drop_cols if c in data.columns]

data = data.dropna(subset=["model_id", target_col]).copy()

unique_models = data["model_id"].dropna().unique()

train_models, test_models = train_test_split(
    unique_models,
    test_size=0.2,
    random_state=42
)

train_data = data[data["model_id"].isin(train_models)].copy()
test_data = data[data["model_id"].isin(test_models)].copy()

print("\n===== CELL-LINE COLD-START SPLIT =====")
print("Total unique cell lines:", len(unique_models))
print("Train cell lines:", len(train_models))
print("Test cell lines:", len(test_models))
print("Train rows:", train_data.shape)
print("Test rows:", test_data.shape)

overlap = set(train_data["model_id"]).intersection(set(test_data["model_id"]))
print("Overlap between train and test cell lines:", len(overlap))

X_train = train_data.drop(columns=drop_cols)
y_train = train_data[target_col]

X_test = test_data.drop(columns=drop_cols)
y_test = test_data[target_col]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:")
print(categorical_cols)

print("\nNumber of numeric columns:", len(numeric_cols))

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_depth=10,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nTraining cell-line cold-start model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

pearson_corr, pearson_p = pearsonr(y_test, y_pred)

print("\n===== CELL-LINE COLD-START RESULTS =====")
print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))
print("R2  :", round(r2, 4))
print("Pearson:", round(pearson_corr, 4))
print("Pearson p-value:", pearson_p)

results = pd.DataFrame({
    "model_id": test_data["model_id"].values,
    "y_true": y_test.values,
    "y_pred": y_pred,
    "abs_error": np.abs(y_test.values - y_pred)
})

results.to_csv("cell_line_cold_start_predictions.csv", index=False)

summary = pd.DataFrame([{
    "split_type": "cell_line_cold_start",
    "total_unique_cell_lines": len(unique_models),
    "train_cell_lines": len(train_models),
    "test_cell_lines": len(test_models),
    "train_rows": len(train_data),
    "test_rows": len(test_data),
    "overlap_cell_lines": len(overlap),
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2,
    "Pearson": pearson_corr,
    "Pearson_p": pearson_p
}])

summary.to_csv("cell_line_cold_start_summary.csv", index=False)

print("\nSaved:")
print("- cell_line_cold_start_predictions.csv")
print("- cell_line_cold_start_summary.csv")