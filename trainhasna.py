import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

import boto3
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
# ---------------- AWS CLIENTS ----------------
AWS_REGION = "us-east-1"
S3_BUCKET = "water-distribution"
S3_KEY = "testbed/"

session2 = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=AWS_REGION
)
s3 = session2.client("s3")
# ==========================================================
# 📦 CHECK & INSTALL torch_geometric
# ==========================================================
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data, Batch
    print("✅ torch_geometric already installed.")
except ImportError:
    print("Installing torch_geometric...")
    os.system("pip install torch_geometric --quiet")
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data, Batch
def load_s3_file(key):
        return s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_KEY}{key}")["Body"].read()
def load_from_s3(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_KEY}{key}")
    return BytesIO(obj["Body"].read())
# ==========================================================
# 1️⃣ LOAD DATA & SYNTHETIC LEAK
# ==========================================================
df = pd.read_csv(BytesIO(load_s3_file("zone_sensor_big_data.csv")))
df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', dayfirst=True)

# ==========================================================
# 2️⃣ FEATURE ENGINEERING
# ==========================================================
flow_cols = sorted([col for col in df.columns if "node" in col.lower() and "pressure" not in col.lower()], 
                   key=lambda x: int(''.join(filter(str.isdigit, x))))
pressure_cols = [col for col in df.columns if "pressure" in col.lower()]

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["Zone_Total"] = df[flow_cols].sum(axis=1)

for col in flow_cols + pressure_cols:
    df[f"{col}_roll_mean_3"] = df[col].rolling(3).mean()
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag2"] = df[col].shift(2)

df = df.dropna().reset_index(drop=True)

# ==========================================================
# 3️⃣ GRAPH STRUCTURE
# ==========================================================
edges = [(0, 1), (0, 4), (0, 6), (1, 2), (1, 3), (4, 5), (6, 7)]
src = [e[0] for e in edges] + [e[1] for e in edges]
dst = [e[1] for e in edges] + [e[0] for e in edges]
edge_index = torch.tensor([src, dst], dtype=torch.long)
num_nodes = len(flow_cols)

node_positions = torch.tensor([0, 1, 2, 2, 1, 2, 1, 2], dtype=torch.float32)

# ==========================================================
# 4️⃣ NORMALIZATION & BASELINE STATS
# ==========================================================
df_model = df.drop(columns=["timestamp"])
flow_scaler = MinMaxScaler()
pressure_scaler = MinMaxScaler()

train_size = int(len(df_model) * 0.70)
train_df = df_model.iloc[:train_size]

# Fit and Save Scalers
flow_scaler.fit(train_df[flow_cols])
pressure_scaler.fit(train_df[pressure_cols])
buffer = BytesIO()
joblib.dump(flow_scaler, buffer)
buffer.seek(0)

s3.put_object(
    Bucket=S3_BUCKET,
    Key=f"{S3_KEY}flow_scaler.pkl",
    Body=buffer.getvalue()
)


buffer = BytesIO()
joblib.dump(pressure_scaler, buffer)
buffer.seek(0)

s3.put_object(
    Bucket=S3_BUCKET,
    Key=f"{S3_KEY}pressure_scaler.pkl",
    Body=buffer.getvalue()
)

# Apply scaling
df_scaled = df_model.copy()
df_scaled[flow_cols] = flow_scaler.transform(df_model[flow_cols])
df_scaled[pressure_cols] = pressure_scaler.transform(df_model[pressure_cols])

# Calculate and Save Baseline Stats (for inference leak thresholding)
normal_df = df.iloc[:train_size]
flow_std_global = {node: normal_df[node].std() + 1e-6 for node in flow_cols}
buffer = BytesIO()
joblib.dump(flow_std_global, buffer)
buffer.seek(0)
s3.put_object(Bucket=S3_BUCKET, Key=f"{S3_KEY}flow_std_global.pkl", Body=buffer.getvalue())
# ==========================================================
# 5️⃣ CREATE GRAPH SEQUENCES
# ==========================================================
seq_length = 24

def create_graph_sequences(df_scaled, df_orig, seq_length=12):
    X_seq, y_seq = [], []
    for i in range(len(df_scaled) - seq_length):
        graphs = []
        for t in range(seq_length):
            row_scaled = df_scaled.iloc[i + t]
            node_feats = []
            for node in flow_cols:
                feat = [row_scaled[node]]
                match_p = [p for p in pressure_cols if node.lower() in p.lower()]
                feat.append(row_scaled[match_p[0]] if match_p else 0.0)
                feat.append(row_scaled["hour_sin"])
                feat.append(row_scaled["hour_cos"])
                feat.append(row_scaled["is_weekend"])
                node_feats.append(feat)

            x = torch.tensor(node_feats, dtype=torch.float32)
            graphs.append(Data(x=x, edge_index=edge_index))

        target_row = df_scaled.iloc[i + seq_length]
        y = torch.tensor([target_row[c] for c in flow_cols], dtype=torch.float32)
        X_seq.append(graphs)
        y_seq.append(y)
    return X_seq, torch.stack(y_seq)

X_all, y_all = create_graph_sequences(df_scaled, df_scaled, seq_length)

train_end = int(len(X_all) * 0.70)
val_end   = int(len(X_all) * 0.85)

X_train, y_train = X_all[:train_end], y_all[:train_end]
X_val,   y_val   = X_all[train_end:val_end], y_all[train_end:val_end]
X_test,  y_test  = X_all[val_end:], y_all[val_end:]

# ==========================================================
# 6️⃣ DEFINE SPATIO-TEMPORAL GNN MODEL
# ==========================================================
class SpatioTemporalGNN(nn.Module):
    def __init__(self, node_feat_dim=5, gat_hidden=32, gru_hidden=128, num_nodes=8, output_dim=8):
        super(SpatioTemporalGNN, self).__init__()
        self.num_nodes = num_nodes
        self.gru_hidden = gru_hidden
        self.att_pool = nn.Linear(gat_hidden, 1)
        self.node_emb = nn.Embedding(self.num_nodes, node_feat_dim)
        self.pos_emb = nn.Linear(1, node_feat_dim)

        self.gat1 = GATConv(node_feat_dim, gat_hidden, heads=2, concat=True, dropout=0.2)
        self.gat2 = GATConv(gat_hidden * 2, gat_hidden, heads=1, concat=False, dropout=0.2)
        self.gru = nn.GRU(gat_hidden, gru_hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(gru_hidden, output_dim)

    def forward(self, graph_sequences):
        batch_size = len(graph_sequences)
        seq_len = len(graph_sequences[0])
        seq_embeddings = []

        for t in range(seq_len):
            graphs_t = [graph_sequences[b][t] for b in range(batch_size)]
            batch_graph = Batch.from_data_list(graphs_t)
            x, edge_idx = batch_graph.x, batch_graph.edge_index

            num_total_nodes = x.size(0)
            node_ids = torch.arange(self.num_nodes, device=x.device).repeat(batch_size)[:num_total_nodes]
            x = x + self.node_emb(node_ids)

            pos = node_positions.to(x.device).repeat(batch_size).unsqueeze(1)[:num_total_nodes]
            x = x + self.pos_emb(pos)

            h = F.elu(self.gat1(x, edge_idx))
            h = F.elu(self.gat2(h, edge_idx))
            
            h_split = torch.split(h, self.num_nodes)
            pooled_list = []
            for h_i in h_split:
                weights = torch.softmax(self.att_pool(h_i), dim=0)
                pooled_list.append((h_i * weights).sum(dim=0))

            seq_embeddings.append(torch.stack(pooled_list))

        seq_tensor = torch.stack(seq_embeddings, dim=1)
        gru_out, _ = self.gru(seq_tensor)
        return self.fc(self.dropout(gru_out[:, -1, :]))

node_feature_dim = 5
model = SpatioTemporalGNN(node_feat_dim=node_feature_dim, gat_hidden=16, gru_hidden=64)
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ==========================================================
# 7️⃣ TRAIN MODEL
# ==========================================================
def get_batches(X, y, batch_size=32):
    for start in range(0, len(X), batch_size):
        yield X[start:start + batch_size], y[start:start + batch_size]

epochs = 50
batch_size = 32
best_val_loss = float("inf")

print("\nStarting GNN training...")
for epoch in range(epochs):
    model.train()
    train_losses = []
    for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

    scheduler.step(val_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(train_losses):.5f} | Val Loss: {val_loss:.5f}")

print("✅ Training complete. Assets saved.")
buffer = BytesIO()
torch.save(best_model_state, buffer)
buffer.seek(0)

s3.put_object(
    Bucket=S3_BUCKET,
    Key=f"{S3_KEY}best_gnn_model.pth",
    Body=buffer.getvalue()
)

print("✅ Final best model saved to S3")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
# ==========================================================
# 8️⃣ EVALUATION
# ==========================================================
buffer = load_from_s3("best_gnn_model.pth")
model.load_state_dict(torch.load(buffer))
model.eval()
with torch.no_grad():
    test_preds = model(X_test).numpy()

y_test_np = y_test.numpy()
inv_preds = flow_scaler.inverse_transform(test_preds)
inv_true  = flow_scaler.inverse_transform(y_test_np)

print(f"\nGNN Test MAE  (real scale): {np.mean(np.abs(inv_preds - inv_true)):.4f}")
print(f"GNN Test RMSE (real scale): {np.sqrt(np.mean((inv_preds - inv_true) ** 2)):.4f}")

# ==========================================================
# 9️⃣ CALCULATE & SAVE DYNAMIC THRESHOLDS
# ==========================================================
print("\nCalculating dynamic thresholds from validation set...")
model.eval()
with torch.no_grad():
    val_preds_scaled = model(X_val).numpy()

val_preds_real = flow_scaler.inverse_transform(val_preds_scaled)
val_true_real  = flow_scaler.inverse_transform(y_val.numpy())

# Calculate directional errors (True - Predicted)
# For flow, a leak means actual flow is HIGHER than predicted
val_errors = val_true_real - val_preds_real

# Find the 99th percentile of error for each node to use as its threshold
node_thresholds = {}
for j, node in enumerate(flow_cols):
    # We only look at positive errors (where true > predicted)
    positive_errors = val_errors[:, j][val_errors[:, j] > 0]
    
    if len(positive_errors) > 0:
        threshold = np.percentile(positive_errors, 99)
    else:
        # Fallback if the model perfectly over-predicts everything (rare)
        threshold = flow_std_global[node] * 3 
        
    node_thresholds[node] = threshold
    print(f"Threshold for {node}: {threshold:.2f}")

buffer = BytesIO()
joblib.dump(node_thresholds, buffer)
buffer.seek(0)
s3.put_object(Bucket=S3_BUCKET, Key=f"{S3_KEY}node_thresholds.pkl", Body=buffer.getvalue())
print("✅ Dynamic thresholds saved.")
