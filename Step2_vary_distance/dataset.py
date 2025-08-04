import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch.nn.functional as F

class SoundAngleDataset(Dataset):
    def __init__(self, audio_dir, csv_path, transform=None, sample_rate=16000, fixed_length=64000):
        """
        audio_dir: ?????? wav ??????
        csv_path: ?? source_index ? angle_center ? CSV ??
        fixed_length: ??????(??:?)
        """
        self.audio_dir = audio_dir
        self.csv_path = csv_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.fixed_length = fixed_length

        # ?? CSV
        df = pd.read_csv(csv_path)
        self.audio_files = [
            os.path.join(audio_dir, f"source_{i:04d}_multi.wav") for i in df["source_index"]
        ]
        self.targets = df["angle_center"].values.astype("float32")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # ?? waveform: [channels, samples]
        waveform, sr = torchaudio.load(self.audio_files[idx])

        # ?????
        if sr != self.sample_rate:
            raise ValueError(f"Sample rate mismatch: expected {self.sample_rate}, got {sr}")

        # ????
        num_frames = waveform.shape[1]
        if num_frames > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]  # ??
        elif num_frames < self.fixed_length:
            pad_len = self.fixed_length - num_frames
            waveform = F.pad(waveform, (0, pad_len))  # ????

        # ?? transform(???????)
        if self.transform:
            waveform = self.transform(waveform)

        # ???????
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)


        return waveform, torch.tensor(self.targets[idx])
