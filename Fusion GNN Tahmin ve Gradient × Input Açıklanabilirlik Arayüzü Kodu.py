import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

st.set_page_config(
    page_title="Cancer Drug Sensitivity Prediction - Fusion Explainability",
    layout="wide"
)

st.title("🧬 Cancer Drug Sensitivity Prediction Platform")
st.subheader("Fusion End-to-End + Explainability Interface")

project_path = r"C:\Users\zuhir\Desktop\New folder"
os.chdir(project_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FusionGNN(nn.Module):
    def __init__(self, num_drugs, num_targets, num_pathways, num_tabular):
        super().__init__()

        self.conv1 = GCNConv(4, 32)
        self.conv2 = GCNConv(32, 64)

        self.drug_emb = nn.Embedding(num_drugs, 24)
        self.target_emb = nn.Embedding(num_targets, 12)
        self.pathway_emb = nn.Embedding(num_pathways, 8)

        self.tab_fc1 = nn.Linear(num_tabular, 32)
        self.tab_fc2 = nn.Linear(32, 16)

        fusion_dim = 64 + 24 + 12 + 8 + 16
        self.fc1 = nn.Linear(fusion_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.25)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        tab = data.tabular.view(num_graphs, -1)
        tab = self.dropout(F.relu(self.tab_fc1(tab)))
        tab = self.dropout(F.relu(self.tab_fc2(tab)))

        drug_vec = self.drug_emb(data.drug_id_enc.view(-1))
        target_vec = self.target_emb(data.target_enc.view(-1))
        pathway_vec = self.pathway_emb(data.pathway_enc.view(-1))

        z = torch.cat([x, drug_vec, target_vec, pathway_vec, tab], dim=1)
        z = self.dropout(F.relu(self.fc1(z)))
        z = self.dropout(F.relu(self.fc2(z)))
        return self.out(z).view(-1)

@st.cache_data
def load_main_data():
    data = pd.read_csv("baseline_merged_dataset.csv")
    if "DRUG_NAME_x" in data.columns:
        data = data.rename(columns={"DRUG_NAME_x": "DRUG_NAME"})
    if "DRUG_NAME_y" in data.columns:
        data = data.drop(columns=["DRUG_NAME_y"])
    return data

@st.cache_data
def load_graph_data():
    edges = pd.read_csv("ppi_edges_top500.csv")
    nodes = pd.read_csv("gnn_node_map.csv")
    multiomics = pd.read_csv("gnn_multiomics_node_features_long.csv")
    return edges, nodes, multiomics

@st.cache_resource
def prepare_fusion_resources():
    data = load_main_data()
    edges, nodes, multiomics = load_graph_data()

    graph_genes = nodes.sort_values("node_idx")["gene"].tolist()
    gene_to_idx = dict(zip(nodes["gene"], nodes["node_idx"]))

    edge_list = []
    for _, row in edges.iterrows():
        g1 = row["gene1"]
        g2 = row["gene2"]
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i = gene_to_idx[g1]
            j = gene_to_idx[g2]
            edge_list.append([i, j])
            edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    rna_wide = multiomics.pivot(index="model_id", columns="gene", values="rna_expr")
    mut_wide = multiomics.pivot(index="model_id", columns="gene", values="mutation_flag")
    cnv_wide = multiomics.pivot(index="model_id", columns="gene", values="cnv_value")
    prot_wide = multiomics.pivot(index="model_id", columns="gene", values="prot_zscore")

    rna_wide = rna_wide[graph_genes]
    mut_wide = mut_wide[graph_genes]
    cnv_wide = cnv_wide[graph_genes]
    prot_wide = prot_wide[graph_genes]

    needed_cols = [
        "model_id",
        "DRUG_ID",
        "TARGET",
        "TARGET_PATHWAY",
        "LN_IC50",
        "mutation_count",
        "driver_mutation_count",
        "mean_copy_number",
        "amp_count",
        "del_count",
        "mean_protein_intensity",
        "mean_protein_zscore",
    ]
    needed_cols = [c for c in needed_cols if c in data.columns]

    fusion_df = data[needed_cols].copy()
    fusion_df["TARGET"] = fusion_df["TARGET"].fillna("Unknown")
    fusion_df["TARGET_PATHWAY"] = fusion_df["TARGET_PATHWAY"].fillna("Unknown")
    fusion_df = fusion_df.drop_duplicates()

    drug_counts = fusion_df["DRUG_ID"].value_counts()
    keep_drugs = drug_counts[drug_counts >= 200].index.tolist()
    fusion_df = fusion_df[fusion_df["DRUG_ID"].isin(keep_drugs)].copy()

    if len(fusion_df) > 25000:
        fusion_df = fusion_df.sample(n=25000, random_state=42).copy()

    fusion_df = fusion_df[fusion_df["model_id"].isin(rna_wide.index)].copy().reset_index(drop=True)

    drug_values = sorted(fusion_df["DRUG_ID"].astype(str).unique().tolist())
    target_values = sorted(fusion_df["TARGET"].astype(str).unique().tolist())
    pathway_values = sorted(fusion_df["TARGET_PATHWAY"].astype(str).unique().tolist())

    drug_to_idx = {v: i for i, v in enumerate(drug_values)}
    target_to_idx = {v: i for i, v in enumerate(target_values)}
    pathway_to_idx = {v: i for i, v in enumerate(pathway_values)}

    tabular_cols = [
        "mutation_count",
        "driver_mutation_count",
        "mean_copy_number",
        "amp_count",
        "del_count",
        "mean_protein_intensity",
        "mean_protein_zscore",
    ]
    tabular_cols = [c for c in tabular_cols if c in fusion_df.columns]

    train_df = fusion_df.sample(frac=0.8, random_state=42)
    tab_mean = train_df[tabular_cols].mean()
    tab_std = train_df[tabular_cols].std().replace(0, 1.0)

    model = FusionGNN(
        num_drugs=len(drug_values),
        num_targets=len(target_values),
        num_pathways=len(pathway_values),
        num_tabular=len(tabular_cols)
    ).to(device)

    model.load_state_dict(torch.load("best_fusion_end_to_end.pt", map_location=device))
    model.eval()

    return {
        "data": data,
        "edge_index": edge_index,
        "graph_genes": graph_genes,
        "rna_wide": rna_wide,
        "mut_wide": mut_wide,
        "cnv_wide": cnv_wide,
        "prot_wide": prot_wide,
        "drug_to_idx": drug_to_idx,
        "target_to_idx": target_to_idx,
        "pathway_to_idx": pathway_to_idx,
        "tabular_cols": tabular_cols,
        "tab_mean": tab_mean,
        "tab_std": tab_std,
        "model": model
    }

resources = prepare_fusion_resources()
data = resources["data"]

st.sidebar.header("Input Selection")

cell_lines = sorted(data["CELL_LINE_NAME"].dropna().unique().tolist())
selected_cell = st.sidebar.selectbox("Select Cell Line", cell_lines)

cell_filtered_df = data[data["CELL_LINE_NAME"] == selected_cell].copy()
drug_names_for_cell = sorted(cell_filtered_df["DRUG_NAME"].dropna().unique().tolist())
selected_drug = st.sidebar.selectbox("Select Drug", drug_names_for_cell)

predict_button = st.sidebar.button("Predict + Explain")

st.markdown("### Selected Inputs")
c1, c2 = st.columns(2)

with c1:
    st.info(f"Cell Line: {selected_cell}")

with c2:
    st.info(f"Drug: {selected_drug}")

filtered = data[
    (data["CELL_LINE_NAME"] == selected_cell) &
    (data["DRUG_NAME"] == selected_drug)
].copy()

st.markdown("### Sample Information")

if len(filtered) > 0:
    row = filtered.iloc[0].copy()

    m1, m2, m3 = st.columns(3)
    with m1:
        st.write("Model ID")
        st.write(row["model_id"])
    with m2:
        st.write("Target")
        st.write(row.get("TARGET", "Unknown"))
    with m3:
        st.write("Target Pathway")
        st.write(row.get("TARGET_PATHWAY", "Unknown"))
else:
    st.warning("No exact row found for this Cell Line + Drug combination.")

st.markdown("### Prediction and Explainability")

if predict_button:
    if len(filtered) == 0:
        st.error("This Cell Line + Drug pair is not available in the current dataset.")
    else:
        row = filtered.iloc[0].copy()

        model_id = row["model_id"]
        drug_id_str = str(row["DRUG_ID"])
        target_str = str(row.get("TARGET", "Unknown"))
        pathway_str = str(row.get("TARGET_PATHWAY", "Unknown"))

        if model_id not in resources["rna_wide"].index:
            st.error("Selected model_id is not available in graph features.")
        elif drug_id_str not in resources["drug_to_idx"]:
            st.error("Selected drug is outside the fusion model vocabulary.")
        elif target_str not in resources["target_to_idx"]:
            st.error("Selected target is outside the fusion model vocabulary.")
        elif pathway_str not in resources["pathway_to_idx"]:
            st.error("Selected pathway is outside the fusion model vocabulary.")
        else:
            rna_vec = resources["rna_wide"].loc[model_id].values.astype(np.float32)
            mut_vec = resources["mut_wide"].loc[model_id].values.astype(np.float32)
            cnv_vec = resources["cnv_wide"].loc[model_id].values.astype(np.float32)
            prot_vec = resources["prot_wide"].loc[model_id].values.astype(np.float32)

            x_np = np.stack([rna_vec, mut_vec, cnv_vec, prot_vec], axis=1)
            x = torch.tensor(x_np, dtype=torch.float32)

            tabular_cols = resources["tabular_cols"]
            tab_vals = row[tabular_cols].astype(float)
            tab_vals = (tab_vals - resources["tab_mean"]) / resources["tab_std"]
            tab_tensor = torch.tensor(tab_vals.values.astype(np.float32), dtype=torch.float32)

            d = Data(
                x=x,
                edge_index=resources["edge_index"],
                y=torch.tensor([0.0], dtype=torch.float32)
            )

            d.batch = torch.zeros(d.x.shape[0], dtype=torch.long)
            d.drug_id_enc = torch.tensor([resources["drug_to_idx"][drug_id_str]], dtype=torch.long)
            d.target_enc = torch.tensor([resources["target_to_idx"][target_str]], dtype=torch.long)
            d.pathway_enc = torch.tensor([resources["pathway_to_idx"][pathway_str]], dtype=torch.long)
            d.tabular = tab_tensor.unsqueeze(0)
            d = d.to(device)

            d.x = d.x.clone().detach().requires_grad_(True)
            d.tabular = d.tabular.clone().detach().requires_grad_(True)

            resources["model"].zero_grad()

            pred_tensor = resources["model"](d)
            pred = pred_tensor.item()

            pred_tensor.backward()

            st.success("Fusion prediction and explainability completed successfully.")

            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric("Predicted LN_IC50", f"{pred:.4f}")
            with p2:
                st.metric("Model", "Fusion End-to-End + Gradients")
            with p3:
                if "LN_IC50" in row.index:
                    st.metric("Reference LN_IC50", f"{row['LN_IC50']:.4f}")

            st.markdown("### Interpretation")
            if pred < -3:
                st.success("High predicted sensitivity to the selected drug.")
            elif pred < -1:
                st.info("Moderate predicted sensitivity to the selected drug.")
            else:
                st.warning("Low predicted sensitivity / possible resistance.")

            if "LN_IC50" in row.index:
                error = abs(pred - row["LN_IC50"])

                st.markdown("### Prediction Error Analysis")
                e1, e2 = st.columns(2)

                with e1:
                    st.metric("Absolute Error", f"{error:.4f}")

                with e2:
                    if error < 0.5:
                        st.success("High confidence prediction")
                    elif error < 1.5:
                        st.info("Moderate confidence")
                    else:
                        st.warning("Low confidence prediction")

            st.markdown("### Fusion Explainability")

            grad_x = d.x.grad.detach().cpu().numpy()
            input_x = d.x.detach().cpu().numpy()
            grad_tab = d.tabular.grad.detach().cpu().numpy().ravel()
            input_tab = d.tabular.detach().cpu().numpy().ravel()

            gx = np.abs(grad_x * input_x)

            modality_names = ["RNA", "Mutation", "CNV", "Proteomics"]
            modality_scores = gx.sum(axis=0)

            modality_df = pd.DataFrame({
                "Modality": modality_names,
                "Importance": modality_scores
            }).sort_values("Importance", ascending=False)

            st.markdown("#### Modality Importance")
            st.dataframe(modality_df, width="stretch")

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.bar(modality_df["Modality"], modality_df["Importance"])
            ax1.set_ylabel("Importance")
            ax1.set_title("Importance by Omics Modality")
            plt.tight_layout()
            st.pyplot(fig1)

            gene_scores = gx.sum(axis=1)
            genes_df = pd.DataFrame({
                "Gene": resources["graph_genes"],
                "Importance": gene_scores,
                "RNA_score": gx[:, 0],
                "Mutation_score": gx[:, 1],
                "CNV_score": gx[:, 2],
                "Proteomics_score": gx[:, 3],
            }).sort_values("Importance", ascending=False).head(15)

            st.markdown("#### Top Influential Genes")
            st.dataframe(genes_df, width="stretch")

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            plot_genes = genes_df.sort_values("Importance", ascending=True)
            ax2.barh(plot_genes["Gene"], plot_genes["Importance"])
            ax2.set_xlabel("Importance")
            ax2.set_ylabel("Gene")
            ax2.set_title("Top Influential Genes (Gradient × Input)")
            plt.tight_layout()
            st.pyplot(fig2)

            tab_scores = np.abs(grad_tab * input_tab)
            tab_df = pd.DataFrame({
                "Feature": tabular_cols,
                "Importance": tab_scores,
                "Normalized_Value": input_tab
            }).sort_values("Importance", ascending=False)

            st.markdown("#### Tabular Feature Importance")
            st.dataframe(tab_df, width="stretch")

            fig3, ax3 = plt.subplots(figsize=(7, 4))
            plot_tab = tab_df.sort_values("Importance", ascending=True)
            ax3.barh(plot_tab["Feature"], plot_tab["Importance"])
            ax3.set_xlabel("Importance")
            ax3.set_ylabel("Feature")
            ax3.set_title("Tabular Summary Importance")
            plt.tight_layout()
            st.pyplot(fig3)

            top_modality = modality_df.iloc[0]["Modality"]
            top_gene = genes_df.iloc[0]["Gene"]
            top_tab = tab_df.iloc[0]["Feature"]

            st.markdown("### Model Explanation Summary")
            st.write(f"- The most influential omics modality for this prediction is **{top_modality}**.")
            st.write(f"- The most influential graph gene is **{top_gene}**.")
            st.write(f"- The most influential summary-level tabular feature is **{top_tab}**.")
            st.write("- Importance here is computed using **Gradient × Input**, which estimates how strongly each input contributed to the prediction for this specific sample.")

            st.markdown("### Biological Summary")
            st.write(f"- **Cell Line:** {selected_cell}")
            st.write(f"- **Drug:** {selected_drug}")
            st.write(f"- **Target:** {target_str}")
            st.write(f"- **Pathway:** {pathway_str}")
            st.write(f"- **Predicted LN_IC50:** {pred:.4f}")
            if "LN_IC50" in row.index:
                st.write(f"- **Reference LN_IC50 from dataset:** {row['LN_IC50']:.4f}")

            st.markdown("### Omics Snapshot")
            snapshot_cols = [
                "mutation_count",
                "driver_mutation_count",
                "mean_copy_number",
                "amp_count",
                "del_count",
                "mean_protein_intensity",
                "mean_protein_zscore"
            ]
            available_snapshot_cols = [c for c in snapshot_cols if c in row.index]
            if available_snapshot_cols:
                snapshot_df = pd.DataFrame({
                    "Feature": available_snapshot_cols,
                    "Value": [row[c] for c in available_snapshot_cols]
                })
                st.dataframe(snapshot_df, width="stretch")

st.markdown("### Available combinations for selected Cell Line")
st.write(f"Number of available drugs for **{selected_cell}**: {len(drug_names_for_cell)}")

with st.expander("Show available drugs for this Cell Line"):
    st.dataframe(pd.DataFrame({"Available Drugs": drug_names_for_cell}), width="stretch")

with st.expander("Show filtered raw row"):
    if len(filtered) > 0:
        st.dataframe(filtered.head(1), width="stretch")

with st.expander("Show dataset preview"):
    st.dataframe(data.head(20), width="stretch")