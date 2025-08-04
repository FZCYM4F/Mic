import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import SoundAngleDataset
from model import MambaAnglePredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -----------------------------
# CONFIGURATION
# -----------------------------
AUDIO_DIR = "/mainfs/scratch/ym4c23/Mic/dataset/RoomSimulation/result_OneSourceMoving/generated_wav/"
CSV_PATH = "/mainfs/scratch/ym4c23/Mic/dataset/RoomSimulation/result_OneSourceMoving/metadata/angles_with_mics.csv"
BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 1e-3
PATIENCE = 10
CLIP_GRAD_NORM = 1.0


def angular_loss(theta_pred, theta_true):
    return torch.mean(1 - torch.cos(theta_pred - theta_true))


class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0


def main():
    # -----------------------------
    # DDP 初始化
    # -----------------------------
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Data
    # -----------------------------
    dataset = SoundAngleDataset(audio_dir=AUDIO_DIR, csv_path=CSV_PATH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(sampler is None), sampler=sampler)

    # -----------------------------
    # Model
    # -----------------------------
    model = MambaAnglePredictor().to(device)
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # -----------------------------
    # Optimizer, Scheduler
    # -----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=(rank == 0))
    early_stopper = EarlyStopping(patience=PATIENCE)

    # -----------------------------
    # Training Loop
    # -----------------------------
    if rank == 0:
        print("Device used:", device)

    for epoch in range(EPOCHS):
        if is_distributed:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        for batch in dataloader:
            waveform, angle = batch
            waveform = waveform.to(device)
            angle = angle.to(device)

            waveform = waveform.permute(0, 2, 1)  # (B, N, 4)

            angle_pred = model(waveform)  # (B,)
            loss = angular_loss(angle_pred, angle)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        if rank == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} - Loss: {avg_loss:.4f}")
            early_stopper(avg_loss, model.module if is_distributed else model)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save model (only rank 0)
    if rank == 0:
        save_path = "mamba_angle_model.pt"
        best_state = early_stopper.best_model_state or model.state_dict()
        torch.save(best_state, save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
