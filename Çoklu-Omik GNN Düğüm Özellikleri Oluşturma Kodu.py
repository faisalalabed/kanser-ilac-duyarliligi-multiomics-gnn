import os
import pandas as pd
import numpy as np

project_path = r"C:\Users\zuhir\Desktop\New folder"
os.chdir(project_path)

print("Current folder:", os.getcwd())

ppi_nodes = pd.read_csv("ppi_nodes_top500.csv")
rna_graph = pd.read_csv("gnn_rna_node_features_wide.csv")
mut = pd.read_csv("mutations_summary_20260316.csv")
cnv = pd.read_csv("cnv_summary_20260316.csv")
prot = pd.read_csv("proteomics_all_20250211.csv")

print("ppi_nodes:", ppi_nodes.shape)
print("rna_graph:", rna_graph.shape)
print("mut:", mut.shape)
print("cnv:", cnv.shape)
print("prot:", prot.shape)

graph_genes = sorted(ppi_nodes["preferred_name"].dropna().unique().tolist())
valid_models = sorted(rna_graph["model_id"].dropna().unique().tolist())

print("\nNumber of graph genes:", len(graph_genes))
print("Number of valid models:", len(valid_models))

rna_long = rna_graph.melt(
    id_vars="model_id",
    var_name="gene",
    value_name="rna_expr"
)

rna_long = rna_long[rna_long["gene"].isin(graph_genes)].copy()
rna_long = rna_long[rna_long["model_id"].isin(valid_models)].copy()

print("\nrna_long:", rna_long.shape)

mut_small = mut[["model_id", "gene_symbol"]].copy()
mut_small = mut_small.dropna()
mut_small = mut_small.rename(columns={"gene_symbol": "gene"})
mut_small = mut_small[mut_small["gene"].isin(graph_genes)].copy()
mut_small = mut_small[mut_small["model_id"].isin(valid_models)].copy()

mut_small["mutation_flag"] = 1
mut_small = mut_small.drop_duplicates(subset=["model_id", "gene"])

print("mut_small:", mut_small.shape)

cnv_small = cnv[["model_id", "symbol", "total_copy_number"]].copy()
cnv_small = cnv_small.dropna()
cnv_small = cnv_small.rename(columns={"symbol": "gene", "total_copy_number": "cnv_value"})
cnv_small = cnv_small[cnv_small["gene"].isin(graph_genes)].copy()
cnv_small = cnv_small[cnv_small["model_id"].isin(valid_models)].copy()

cnv_small = (
    cnv_small.groupby(["model_id", "gene"], as_index=False)["cnv_value"]
    .mean()
)

print("cnv_small:", cnv_small.shape)

prot_small = prot[["model_id", "symbol", "zscore"]].copy()
prot_small = prot_small.dropna()
prot_small = prot_small.rename(columns={"symbol": "gene", "zscore": "prot_zscore"})
prot_small = prot_small[prot_small["gene"].isin(graph_genes)].copy()
prot_small = prot_small[prot_small["model_id"].isin(valid_models)].copy()

prot_small = (
    prot_small.groupby(["model_id", "gene"], as_index=False)["prot_zscore"]
    .mean()
)

print("prot_small:", prot_small.shape)

full_index = pd.MultiIndex.from_product(
    [valid_models, graph_genes],
    names=["model_id", "gene"]
)

full_df = pd.DataFrame(index=full_index).reset_index()
print("\nfull_df:", full_df.shape)

multiomics = full_df.merge(rna_long, on=["model_id", "gene"], how="left")
multiomics = multiomics.merge(mut_small, on=["model_id", "gene"], how="left")
multiomics = multiomics.merge(cnv_small, on=["model_id", "gene"], how="left")
multiomics = multiomics.merge(prot_small, on=["model_id", "gene"], how="left")

print("multiomics before fill:", multiomics.shape)
print(multiomics.head())

multiomics["mutation_flag"] = multiomics["mutation_flag"].fillna(0)
multiomics["cnv_value"] = multiomics["cnv_value"].fillna(2.0)
multiomics["prot_zscore"] = multiomics["prot_zscore"].fillna(0.0)

multiomics["rna_expr"] = multiomics["rna_expr"].fillna(0.0)

print("\nMissing values after fill:")
print(multiomics[["rna_expr", "mutation_flag", "cnv_value", "prot_zscore"]].isna().sum())

multiomics.to_csv("gnn_multiomics_node_features_long.csv", index=False)

rna_wide = multiomics.pivot(index="model_id", columns="gene", values="rna_expr").reset_index()
mut_wide = multiomics.pivot(index="model_id", columns="gene", values="mutation_flag").reset_index()
cnv_wide = multiomics.pivot(index="model_id", columns="gene", values="cnv_value").reset_index()
prot_wide = multiomics.pivot(index="model_id", columns="gene", values="prot_zscore").reset_index()

rna_wide.to_csv("check_rna_wide_439.csv", index=False)
mut_wide.to_csv("check_mut_wide_439.csv", index=False)
cnv_wide.to_csv("check_cnv_wide_439.csv", index=False)
prot_wide.to_csv("check_prot_wide_439.csv", index=False)

print("\nSaved:")
print("- gnn_multiomics_node_features_long.csv")
print("- check_rna_wide_439.csv")
print("- check_mut_wide_439.csv")
print("- check_cnv_wide_439.csv")
print("- check_prot_wide_439.csv")