# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader
from dataset import SoundAngleDataset
from model import MambaAnglePredictor

# -----------------------------
# CONFIGURATION
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mamba_angle_model.pt"

# ✅ 替换成你实际的数据路径（评估集）
AUDIO_DIR = "/scratch/ym4c23/Mic/Step3_/result/evalminiaudio"
CSV_PATH = "/scratch/ym4c23/Mic/Step3_/result/evalminiCSV/angles_with_mics.csv"
BATCH_SIZE = 8

# -----------------------------
# LOSS FUNCTION
# -----------------------------
def angular_loss(theta_pred, theta_true):
    return torch.mean(1 - torch.cos(theta_pred - theta_true))

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = SoundAngleDataset(audio_dir=AUDIO_DIR, csv_path=CSV_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = MambaAnglePredictor()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# EVALUATION
# -----------------------------
total_loss = 0.0
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in dataloader:
        # ✅ 加载 waveform, mic_positions, angle
        waveform, mic_positions, angle = batch
        waveform = waveform.to(DEVICE)            # (B, 4, N)
        mic_positions = mic_positions.to(DEVICE)  # (B, 4, 3)
        angle = angle.to(DEVICE)

        waveform = waveform.permute(0, 2, 1)       # (B, N, 4)
        mic_positions = mic_positions.view(waveform.size(0), -1)  # ✅ 展平为 (B, 12)

        angle_pred = model(waveform, mic_positions)  # ✅ 加入 mic_pos
        angle_pred = (angle_pred + torch.pi) % (2 * torch.pi) - torch.pi  # wrap 到 [-pi, pi]

        loss = angular_loss(angle_pred, angle)
        total_loss += loss.item()

        all_preds.extend(angle_pred.cpu().numpy())
        all_targets.extend(angle.cpu().numpy())

avg_loss = total_loss / len(dataloader)
print(f"angular loss: {avg_loss:.4f}\n")

# ✅ 可选：显示部分预测值
print("True angles:", [round(a, 3) for a in all_targets[10:15]])
print("Pred angles:", [round(p, 3) for p in all_preds[10:15]])
