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

# �滻����Ĳ��Լ�·��
AUDIO_DIR = "/scratch/ym4c23/Mic/Step2_vary_distance/result/evalminiaudio"
CSV_PATH = "/scratch/ym4c23/Mic/Step2_vary_distance/result/evalminiCSV/angles_with_mics.csv"
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
        waveform, angle = batch
        waveform = waveform.to(DEVICE)  # (B, 4, N)
        angle = angle.to(DEVICE)

        waveform = waveform.permute(0, 2, 1)  # (B, N, 4)
        angle_pred = model(waveform)         # (B,)
        angle_pred = (angle_pred + torch.pi) % (2 * torch.pi) - torch.pi  # wrap ? [-p, p]

        loss = angular_loss(angle_pred, angle)
        total_loss += loss.item()

        all_preds.extend(angle_pred.cpu().numpy())
        all_targets.extend(angle.cpu().numpy())

avg_loss = total_loss / len(dataloader)
print(f"angular loss: {avg_loss:.4f}\n")

# ��ѡ����ʾǰ����Ԥ��
print("True angles:", [round(a, 3) for a in all_targets[:5]])
print("Pred angles:", [round(p, 3) for p in all_preds[:5]])
