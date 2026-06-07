import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

project_path = r"C:\Users\zuhir\Desktop\New folder"
os.chdir(project_path)

print("Current folder:", os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

edges = pd.read_csv("ppi_edges_top500.csv")
nodes = pd.read_csv("gnn_node_map.csv")
multiomics = pd.read_csv("gnn_multiomics_node_features_long.csv")
data = pd.read_csv("baseline_merged_dataset.csv")

print("edges:", edges.shape)
print("nodes:", nodes.shape)
print("multiomics:", multiomics.shape)
print("data:", data.shape)

if "DRUG_NAME_x" in data.columns:
    data = data.rename(columns={"DRUG_NAME_x": "DRUG_NAME"})
if "DRUG_NAME_y" in data.columns:
    data = data.drop(columns=["DRUG_NAME_y"])

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

print("\nfusion_df before filtering:", fusion_df.shape)

drug_counts = fusion_df["DRUG_ID"].value_counts()
keep_drugs = drug_counts[drug_counts >= 200].index.tolist()
fusion_df = fusion_df[fusion_df["DRUG_ID"].isin(keep_drugs)].copy()

max_rows = 25000
if len(fusion_df) > max_rows:
    fusion_df = fusion_df.sample(n=max_rows, random_state=42).copy()

fusion_df = fusion_df.reset_index(drop=True)

print("fusion_df after drug filtering/cap:", fusion_df.shape)
print("Unique model_id:", fusion_df["model_id"].nunique())
print("Unique drugs:", fusion_df["DRUG_ID"].nunique())

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
print("edge_index shape:", edge_index.shape)

graph_genes = nodes.sort_values("node_idx")["gene"].tolist()

rna_wide = multiomics.pivot(index="model_id", columns="gene", values="rna_expr")
mut_wide = multiomics.pivot(index="model_id", columns="gene", values="mutation_flag")
cnv_wide = multiomics.pivot(index="model_id", columns="gene", values="cnv_value")
prot_wide = multiomics.pivot(index="model_id", columns="gene", values="prot_zscore")

rna_wide = rna_wide[graph_genes]
mut_wide = mut_wide[graph_genes]
cnv_wide = cnv_wide[graph_genes]
prot_wide = prot_wide[graph_genes]

valid_models = set(rna_wide.index)
fusion_df = fusion_df[fusion_df["model_id"].isin(valid_models)].copy()
fusion_df = fusion_df.reset_index(drop=True)

print("fusion_df after multiomics filter:", fusion_df.shape)

le_drug = LabelEncoder()
le_target = LabelEncoder()
le_pathway = LabelEncoder()

fusion_df["DRUG_ID_enc"] = le_drug.fit_transform(fusion_df["DRUG_ID"].astype(str))
fusion_df["TARGET_enc"] = le_target.fit_transform(fusion_df["TARGET"].astype(str))
fusion_df["PATHWAY_enc"] = le_pathway.fit_transform(fusion_df["TARGET_PATHWAY"].astype(str))

num_drugs = fusion_df["DRUG_ID_enc"].nunique()
num_targets = fusion_df["TARGET_enc"].nunique()
num_pathways = fusion_df["PATHWAY_enc"].nunique()

print("Encoded drugs:", num_drugs)
print("Encoded targets:", num_targets)
print("Encoded pathways:", num_pathways)

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

print("Tabular numeric columns:", tabular_cols)

for c in tabular_cols:
    fusion_df[c] = fusion_df[c].fillna(fusion_df[c].median())

train_df, test_df = train_test_split(
    fusion_df,
    test_size=0.2,
    random_state=42
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nTrain rows:", train_df.shape)
print("Test rows :", test_df.shape)

tab_mean = train_df[tabular_cols].mean()
tab_std = train_df[tabular_cols].std().replace(0, 1.0)

train_df.loc[:, tabular_cols] = (train_df[tabular_cols] - tab_mean) / tab_std
test_df.loc[:, tabular_cols] = (test_df[tabular_cols] - tab_mean) / tab_std

def build_dataset(df):
    data_list = []

    for _, row in df.iterrows():
        model_id = row["model_id"]

        rna_vec = rna_wide.loc[model_id].values.astype(np.float32)
        mut_vec = mut_wide.loc[model_id].values.astype(np.float32)
        cnv_vec = cnv_wide.loc[model_id].values.astype(np.float32)
        prot_vec = prot_wide.loc[model_id].values.astype(np.float32)

        x = np.stack([rna_vec, mut_vec, cnv_vec, prot_vec], axis=1)
        x = torch.tensor(x, dtype=torch.float32)

        tab_vec = row[tabular_cols].values.astype(np.float32)
        tab_vec = torch.tensor(tab_vec, dtype=torch.float32)

        d = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([row["LN_IC50"]], dtype=torch.float32)
        )

        d.drug_id_enc = torch.tensor([row["DRUG_ID_enc"]], dtype=torch.long)
        d.target_enc = torch.tensor([row["TARGET_enc"]], dtype=torch.long)
        d.pathway_enc = torch.tensor([row["PATHWAY_enc"]], dtype=torch.long)
        d.tabular = tab_vec

        data_list.append(d)

    return data_list

train_data = build_dataset(train_df)
test_data = build_dataset(test_df)

print("Number of train graphs:", len(train_data))
print("Number of test graphs:", len(test_data))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

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

        tab = data.tabular.view(data.num_graphs, -1)
        tab = self.dropout(F.relu(self.tab_fc1(tab)))
        tab = self.dropout(F.relu(self.tab_fc2(tab)))

        drug_vec = self.drug_emb(data.drug_id_enc.view(-1))
        target_vec = self.target_emb(data.target_enc.view(-1))
        pathway_vec = self.pathway_emb(data.pathway_enc.view(-1))

        z = torch.cat([x, drug_vec, target_vec, pathway_vec, tab], dim=1)
        z = self.dropout(F.relu(self.fc1(z)))
        z = self.dropout(F.relu(self.fc2(z)))
        return self.out(z).view(-1)

model = FusionGNN(
    num_drugs=num_drugs,
    num_targets=num_targets,
    num_pathways=num_pathways,
    num_tabular=len(tabular_cols)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def train_epoch():
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        loss = F.mse_loss(pred, batch.y.view(-1))

        loss.backward()
        optimizer.step()

        bs = batch.num_graphs
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / total_n

def eval_epoch(loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)

            preds.extend(pred.cpu().numpy().tolist())
            trues.extend(batch.y.view(-1).cpu().numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)

    return rmse, mae, r2, preds, trues

best_r2 = -999
best_metrics = None
best_preds = None
best_trues = None

print("\nTraining Fusion end-to-end model...")

for epoch in range(1, 21):
    train_loss = train_epoch()
    rmse, mae, r2, preds, trues = eval_epoch(test_loader)

    if r2 > best_r2:
        best_r2 = r2
        best_metrics = (rmse, mae, r2)
        best_preds = preds
        best_trues = trues
        torch.save(model.state_dict(), "best_fusion_end_to_end.pt")

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test RMSE: {rmse:.4f} | "
        f"Test MAE: {mae:.4f} | "
        f"Test R2: {r2:.4f}"
    )

results_df = pd.DataFrame({
    "y_true": best_trues,
    "y_pred": best_preds
})
results_df.to_csv("fusion_end_to_end_predictions.csv", index=False)

print("\nBest Fusion End-to-End Results")
print("RMSE:", round(best_metrics[0], 4))
print("MAE :", round(best_metrics[1], 4))
print("R2  :", round(best_metrics[2], 4))

print("\nSaved:")
print("- best_fusion_end_to_end.pt")
print("- fusion_end_to_end_predictions.csv")