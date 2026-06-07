import os
import pandas as pd

project_path = r"C:\Users\zuhir\Desktop\New folder"
os.chdir(project_path)

print("Current folder:", os.getcwd())

response = pd.read_csv("core_response.csv")
drug_features = pd.read_csv("core_drug_features.csv")
mutation_summary = pd.read_csv("core_mutation_summary.csv")
cnv_summary = pd.read_csv("core_cnv_summary.csv")
proteomics_summary = pd.read_csv("core_proteomics_summary.csv")
availability = pd.read_csv("core_availability.csv")
rna_top = pd.read_csv("core_rna_top500.csv")

print("\nLoaded files:")
print("response:", response.shape)
print("drug_features:", drug_features.shape)
print("mutation_summary:", mutation_summary.shape)
print("cnv_summary:", cnv_summary.shape)
print("proteomics_summary:", proteomics_summary.shape)
print("availability:", availability.shape)
print("rna_top:", rna_top.shape)

avail_filtered = availability[
    (availability["RNASeq Sanger Cell Lines"] == True) &
    (availability["Mutation Sanger Cell Lines (WES)"] == True) &
    (availability["CNV Sanger Cell Lines (WES)"] == True)
].copy()

print("\navail_filtered shape:", avail_filtered.shape)
print(avail_filtered.head())

valid_models = set(avail_filtered["model_id"])

response_f = response[response["model_id"].isin(valid_models)].copy()
mutation_f = mutation_summary[mutation_summary["model_id"].isin(valid_models)].copy()
cnv_f = cnv_summary[cnv_summary["model_id"].isin(valid_models)].copy()
prot_f = proteomics_summary[proteomics_summary["model_id"].isin(valid_models)].copy()
rna_f = rna_top[rna_top["model_id"].isin(valid_models)].copy()

print("\nAfter filtering by availability:")
print("response_f:", response_f.shape)
print("mutation_f:", mutation_f.shape)
print("cnv_f:", cnv_f.shape)
print("prot_f:", prot_f.shape)
print("rna_f:", rna_f.shape)

data = response_f.merge(drug_features, on="DRUG_ID", how="left")
print("\nAfter merge with drug_features:", data.shape)

data = data.merge(mutation_f, on="model_id", how="inner")
print("After merge with mutation_summary:", data.shape)

data = data.merge(cnv_f, on="model_id", how="inner")
print("After merge with cnv_summary:", data.shape)

data = data.merge(prot_f, on="model_id", how="inner")
print("After merge with proteomics_summary:", data.shape)

data = data.merge(rna_f, on="model_id", how="inner")
print("After merge with rna_top500:", data.shape)

data = data.drop_duplicates()
print("\nAfter drop_duplicates:", data.shape)

missing_before = data.isna().sum().sum()
print("Total missing values before fill/drop:", missing_before)

for col in ["TARGET", "TARGET_PATHWAY", "DRUG_NAME"]:
    if col in data.columns:
        data[col] = data[col].fillna("Unknown")

data = data.dropna()

print("After dropna:", data.shape)

print("\nUnique model_id:", data["model_id"].nunique())
print("Unique drugs:", data["DRUG_ID"].nunique())

if "CELL_LINE_NAME" in data.columns:
    print("Unique cell line names:", data["CELL_LINE_NAME"].nunique())

print("\nFirst 5 rows:")
print(data.head())

data.to_csv("baseline_merged_dataset.csv", index=False)

print("\nSaved: baseline_merged_dataset.csv")