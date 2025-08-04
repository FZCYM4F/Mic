import torch
import torch.nn as nn
from mamba_block.mamba2_simple import Mamba2Simple 

class MambaAnglePredictor(nn.Module):
    def __init__(self, input_dim=4, d_model=192, d_mamba=64, mix_dim=32):
        """
        input_dim: Number of input channels (e.g., 4 microphones)
        d_model: Dimension for Mamba input/output
        d_mamba: Internal state dimension of Mamba
        mix_dim: Dimension for mixing features before final output
        """
        super(MambaAnglePredictor, self).__init__()

        # First linear layer to expand input features
        self.input_fc = nn.Linear(input_dim, d_model)

        # Mamba2 backbone (sequence modeling)
        self.mamba = Mamba2Simple(d_model=d_model, d_state=d_mamba, expand=2, headdim=48)

        # Mixing layer to reduce dimensionality after Mamba
        # self.mix = nn.Linear(d_model, mix_dim)
        self.mix = nn.Sequential(
            nn.Linear(d_model, mix_dim),
            nn.ReLU(),
            nn.Linear(mix_dim, mix_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )


        # Final linear layer to predict one angle value (in radians)
        self.output_fc = nn.Linear(mix_dim, 1)

        nn.init.zeros_(self.output_fc.bias)
        nn.init.kaiming_uniform_(self.output_fc.weight, a=0.01)


    def forward(self, x):
        """
        x: Tensor of shape (B, N, C)
           B = batch size
           N = number of samples in audio sequence
           C = number of channels (e.g., 4 microphones)
        """
        # Rearrange input to (B, N, C)
        # x is expected to be in shape (batch, seq_len, channels)
        x = self.input_fc(x)              # (B, N, d_model)
        x = self.mamba(x)                 # (B, N, d_model)
        x = self.mix(x)                   # (B, N, mix_dim)

        # Temporal average pooling over time (sequence length)
        x = torch.mean(x, dim=1)          # (B, mix_dim)

        # Final angle prediction
        angle = self.output_fc(x)         # (B, 1)
        # print("?? Raw output before tanh:", angle.mean().detach().item(), angle.std().detach().item())
        # angle = torch.pi * torch.tanh(angle)  # Map to (-p, p)
        return angle.squeeze(1)           # (B,)
    

