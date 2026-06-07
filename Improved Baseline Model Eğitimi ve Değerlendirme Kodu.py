import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

project_path = r"C:\Users\zuhir\Desktop\New folder"
os.chdir(project_path)

print("Current folder:", os.getcwd())

data = pd.read_csv("baseline_merged_dataset.csv")
print("Original data shape:", data.shape)

if "DRUG_NAME_x" in data.columns:
    data = data.rename(columns={"DRUG_NAME_x": "DRUG_NAME"})
if "DRUG_NAME_y" in data.columns:
    data = data.drop(columns=["DRUG_NAME_y"])

print("After cleaning duplicate columns:", data.shape)

drop_cols = [
    "LN_IC50",
    "CELL_LINE_NAME",
    "model_id",
    "DRUG_NAME"
]

drop_cols = [c for c in drop_cols if c in data.columns]

X = data.drop(columns=drop_cols)
y = data["LN_IC50"]

print("X shape:", X.shape)
print("y shape:", y.shape)

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTraining improved baseline...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nImproved Baseline Results")
print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))
print("R2  :", round(r2, 4))

results = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred
})
results.to_csv("improved_baseline_predictions.csv", index=False)

print("\nSaved: improved_baseline_predictions.csv")