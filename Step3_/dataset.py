import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch.nn.functional as F
import numpy as np

class SoundAngleDataset(Dataset):
    def __init__(self, audio_dir, csv_path, transform=None, sample_rate=16000, fixed_length=64000):
        """
        audio_dir: 存放多通道 .wav 文件的文件夹
        csv_path: 包含 source_index 和 angle_center 等元数据的 CSV 文件
        fixed_length: 固定音频长度（样本数）
        """
        self.audio_dir = audio_dir
        self.csv_path = csv_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.fixed_length = fixed_length

        # 加载 CSV 文件
        self.df = pd.read_csv(csv_path)

        # 构建音频路径列表
        self.audio_files = [
            os.path.join(audio_dir, f"source_{i:05d}_multi.wav") for i in self.df["source_index"]
        ]
        self.targets = self.df["angle_relative_to_head"].values.astype("float32")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # 加载波形: [channels, samples]
        waveform, sr = torchaudio.load(self.audio_files[idx])

        if sr != self.sample_rate:
            raise ValueError(f"Sample rate mismatch: expected {self.sample_rate}, got {sr}")

        # 裁剪或填充音频长度
        num_frames = waveform.shape[1]
        if num_frames > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]
        elif num_frames < self.fixed_length:
            pad_len = self.fixed_length - num_frames
            waveform = F.pad(waveform, (0, pad_len))

        # 归一化波形
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        # 获取目标角度
        angle = torch.tensor(self.targets[idx])

        # 提取麦克风位置 (4 x 3)
        row = self.df.iloc[idx]
        mic_positions = np.array([
            [row["mic_L_x"], row["mic_L_y"], row["mic_L_z"]],
            [row["mic_R_x"], row["mic_R_y"], row["mic_R_z"]],
            [row["mic_C_x"], row["mic_C_y"], row["mic_C_z"]],
            [row["mic_T_x"], row["mic_T_y"], row["mic_T_z"]],  # 如没有 T，可注释本行
        ], dtype=np.float32)

        return waveform, mic_positions, angle
