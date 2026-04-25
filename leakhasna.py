from io import StringIO

from joblib.numpy_pickle_compat import BytesIO
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from datetime import timedelta
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
import os
import argparse
import sys
import boto3
from dotenv import load_dotenv
load_dotenv()
AWS_REGION = "us-east-1"
S3_BUCKET = "water-distribution"
S3_KEY = "testbed/"



# ---------------- AWS CLIENTS ----------------

session2 = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=AWS_REGION
)
s3 = session2.client("s3")
# ==========================================================
# 1️⃣ ARCHITECTURE RE-DEFINITION
# ==========================================================
# (Must match the architecture used during training exactly)
edges = [(0, 1), (0, 4), (0, 6), (1, 2), (1, 3), (4, 5), (6, 7)]
src = [e[0] for e in edges] + [e[1] for e in edges]
dst = [e[1] for e in edges] + [e[0] for e in edges]
edge_index = torch.tensor([src, dst], dtype=torch.long)
node_positions = torch.tensor([0, 1, 2, 2, 1, 2, 1, 2], dtype=torch.float32)
def load_s3_file(key):
        return s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_KEY}{key}")["Body"].read()
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

# ==========================================================
# 2️⃣ LOAD MODEL & ASSETS
# ==========================================================
print("Loading model and preprocessing assets...")

flow_scaler = joblib.load(BytesIO(load_s3_file("flow_scaler.pkl")))
pressure_scaler = joblib.load(BytesIO(load_s3_file("pressure_scaler.pkl")))
flow_std_global = joblib.load(BytesIO(load_s3_file("flow_std_global.pkl")))

model = SpatioTemporalGNN(node_feat_dim=5, gat_hidden=16, gru_hidden=64)
model.load_state_dict(torch.load(BytesIO(load_s3_file("best_gnn_model.pth")), map_location="cpu"))
model.eval()

# ==========================================================
# 3️⃣ LOAD & PREPARE DATA
# ==========================================================

df = pd.read_csv(BytesIO(load_s3_file("zone_sensor_big_data.csv")))
df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', dayfirst=True)

# Feature Engineering
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

# Apply saved scalers
df_model = df.drop(columns=["timestamp"])
df_scaled = df_model.copy()
df_scaled[flow_cols] = flow_scaler.transform(df_model[flow_cols])
df_scaled[pressure_cols] = pressure_scaler.transform(df_model[pressure_cols])

# ==========================================================
# 4️⃣ GRAPH SEQUENCE GENERATOR
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

# ==========================================================
# 5️⃣ ROBUST LEAK DETECTION (Directional + Persistent)
# ==========================================================
print("\nRunning robust leak detection...")

# Load the dynamic thresholds calculated during training
node_thresholds = joblib.load("node_thresholds.pkl")
timestamps = df["timestamp"].iloc[seq_length:].reset_index(drop=True)
leak_results = []

model.eval()
with torch.no_grad():
    for i in range(len(X_all)):
        pred = model([X_all[i]]).numpy()
        true = y_all[i].numpy().reshape(1, -1)

        pred_real = flow_scaler.inverse_transform(pred)
        true_real = flow_scaler.inverse_transform(true)

        node_flags = []

        for j, node in enumerate(flow_cols):
            # 1. Physics-Informed Direction: 
            # If it's a leak, the actual flow should be HIGHER than the model predicts.
            flow_residual = true_real[0][j] - pred_real[0][j]
            print(f"Node: {node}, Predicted: {pred_real[0][j]:.2f}, Actual: {true_real[0][j]:.2f}, Residual: {flow_residual:.2f}")
            # 2. Dynamic Thresholding:
            # Does the positive residual exceed this specific node's 99th percentile error?
            if flow_residual >  0.5:
                node_flags.append(1)
            else:
                node_flags.append(0)

        leak_results.append(node_flags)

# Create DataFrame
leak_df = pd.DataFrame(leak_results, columns=[f"{n}_Raw_Flag" for n in flow_cols])
leak_df.insert(0, "Timestamp", timestamps)

# 3. Persistence Filter (Reduce False Positives)
# A leak must be flagged for 3 consecutive timesteps to trigger an alarm
rolling_window = 1

for node in flow_cols:
    raw_col = f"{node}_Raw_Flag"
    alarm_col = f"{node}_CONFIRMED_ALARM"
    
    # Sum the flags over the last 3 timesteps
    rolling_sum = leak_df[raw_col].rolling(window=rolling_window, min_periods=1).sum()
    
    # If the sum equals the window size, it means all recent timesteps were flagged
    leak_df[alarm_col] = (rolling_sum >= rolling_window).astype(int)

# Save the full report
# leak_df.to_csv("final_leak_report.csv", index=False)

# ==================================================
# CURRENT LEAK STATUS USING LAST 3 ROWS
# ==================================================
alarm_cols = [col for col in leak_df.columns if "CONFIRMED_ALARM" in col]

# Take last 3 rows
recent_rows = leak_df.tail(3)

# Sum alarms in last 3 rows
node_alarm_sum = recent_rows[alarm_cols].sum()

# Node leak decision (2 out of last 3 rows)
current_nodes = {}
for col in alarm_cols:
    current_nodes[col] = 1 if node_alarm_sum[col] >= 2 else 0

# Overall leak now
leak_now = 1 if any(v == 1 for v in current_nodes.values()) else 0

# Save readable dataframe
leaks_only = pd.DataFrame([{
    "Timestamp": recent_rows["Timestamp"].iloc[-1],
    "Leak_Now": leak_now,
    **current_nodes
}])
# leaks_only.to_csv("detected_leaks_only.csv", index=False)
csv_buffer = StringIO()
leaks_only.to_csv(csv_buffer, index=False)

s3.put_object(
    Bucket=S3_BUCKET,
    Key=f"{S3_KEY}detected_leaks_only.csv",
    Body=csv_buffer.getvalue(),
    ContentType="text/csv"
)

print("detected_leaks_only.csv uploaded successfully to S3")
print(f"✅ Leak detection complete. Found {len(leaks_only)} confirmed anomaly events.")

# ==========================================================
# 6️⃣ NEXT HOUR FORECAST WITH MC DROPOUT
# ==========================================================
print("\nGenerating next hour forecast...")

# Build last sequence
last_graphs = []
for t in range(-seq_length, 0):
    row_scaled = df_scaled.iloc[t]
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
    last_graphs.append(Data(x=x, edge_index=edge_index))

def mc_dropout_gnn(model, graph_seq, samples=100):
    model.train()  # Keep dropout active
    preds = []
    for _ in range(samples):
        with torch.no_grad():
            preds.append(model([graph_seq]).numpy())
    preds = np.array(preds) 
    return preds[:, 0, :].mean(axis=0), preds[:, 0, :].std(axis=0)

mean_pred, std_pred = mc_dropout_gnn(model, last_graphs)

forecast = flow_scaler.inverse_transform(mean_pred.reshape(1, -1))
confidence = np.clip(100 - (std_pred * 100), 0, 100)
upper_95 = forecast + 1.96 * std_pred
lower_95 = forecast - 1.96 * std_pred

next_timestamp = df["timestamp"].iloc[-1] + timedelta(hours=1)


# ==========================================================
# NEW PART: Save Only Node2 / Node5 / Node7 Valve + Zone Demand
# ==========================================================
import json

print(flow_cols)

# Build demand map from forecast
node_demand_map = {}
for i, node in enumerate(flow_cols):
    node_demand_map[node] = round(float(forecast[0][i]), 2)

# ----------------------------------------------------------
# Zone Mapping
# Node2 valve controls Node3 + Node4
# Node5 valve controls Node5 + Node6
# Node7 valve controls Node7 + Node8
# ----------------------------------------------------------

zone_2 = node_demand_map[flow_cols[2]] 
zone_5 = node_demand_map[flow_cols[5]] 
zone_7 = node_demand_map[flow_cols[7]] 

zone_demands = {
    "Node2_Flow_Valve": round(zone_2, 2),
    "Node5_Flow_Valve": round(zone_5, 2),
    "Node7_Flow_Valve": round(zone_7, 2)
}

# Highest demand valve ON
best_valve = max(zone_demands, key=zone_demands.get)

output = {
    "timestamp": str(next_timestamp),
    "valves": {
        "Node2_Flow_Valve": 1 if best_valve == "Node2_Flow_Valve" else 0,
        "Node5_Flow_Valve": 1 if best_valve == "Node5_Flow_Valve" else 0,
        "Node7_Flow_Valve": 1 if best_valve == "Node7_Flow_Valve" else 0
    },
    "demand": {
        "Node2_Zone_Demand": round(zone_2, 2),
        "Node5_Zone_Demand": round(zone_5, 2),
        "Node7_Zone_Demand": round(zone_7, 2)
    }
}

# with open("node_valve_demand.json", "w") as f:
#     json.dump(output, f, indent=4)
# Convert JSON to string
json_data = json.dumps(output, indent=4)

# Upload to S3
s3.put_object(
    Bucket=S3_BUCKET,
    Key=f"{S3_KEY}node_valve_demand.json",
    Body=json_data,
    ContentType="application/json"
)

print("✅ node_valve_demand.json uploaded successfully to S3")