import os
import pandas as pd

project_path = r"C:\Users\zuhir\Desktop\New folder"
os.chdir(project_path)

print("Current folder:", os.getcwd())
print("\nFiles in folder:")
for f in os.listdir():
    print("-", f)

print("\nReading files...\n")

# 1) GDSC
gdsc = pd.read_excel("GDSC2_fitted_dose_response.xlsx")
print("GDSC:", gdsc.shape)

# 2) Compounds
compounds = pd.read_csv("screened_compounds_rel_8.5 (2).csv")
print("Compounds:", compounds.shape)

# 3) RNA
rna = pd.read_csv("rnaseq_merged_rsem_fpkm_20260323.csv")
print("RNA:", rna.shape)

# 4) Mutations
mut = pd.read_csv("mutations_summary_20260316.csv")
print("Mutations:", mut.shape)

# 5) CNV
cnv = pd.read_csv("cnv_summary_20260316.csv")
print("CNV:", cnv.shape)

# 6) Proteomics
prot = pd.read_csv("proteomics_all_20250211.csv")
print("Proteomics:", prot.shape)

# 7) Gene identifiers
genes = pd.read_csv("gene_identifiers_20241212 (3).csv")
print("Genes:", genes.shape)

# 8) Availability
avail = pd.read_csv("model_dataset_availability_20220315.csv")
print("Availability:", avail.shape)

# 9) PPI links
ppi_links = pd.read_csv("9606.protein.links.v12.0.txt.gz", sep=" ")
print("PPI links:", ppi_links.shape)

# 10) PPI info
ppi_info = pd.read_csv("9606.protein.info.v12.0.txt.gz", sep="\t")
print("PPI info:", ppi_info.shape)