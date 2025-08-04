import torch
import torch.nn as nn
from mamba_block.mamba2_simple import Mamba2Simple


class MambaAnglePredictor(nn.Module):
    def __init__(self, input_dim=4, d_model=192, d_mamba=64, mix_dim=32, mic_pos_dim=12):
        """
        input_dim: 语音输入通道数（4个麦克风）
        mic_pos_dim: 麦克风位置向量长度（4个麦克风，每个3维 = 12）
        d_model: Mamba 输入/输出维度
        d_mamba: Mamba 状态维度
        mix_dim: 中间混合特征维度
        """
        super(MambaAnglePredictor, self).__init__()

        # 🎧 音频输入特征编码器（将4维输入映射到d_model）
        self.input_fc = nn.Linear(input_dim, d_model)

        # 🧭 麦克风位置编码器（将12维位置编码为 d_model）
        self.mic_fc = nn.Sequential(
            nn.Linear(mic_pos_dim, d_model),
            nn.ReLU()
        )

        # 🧠 Mamba主干网络
        self.mamba = Mamba2Simple(d_model=d_model, d_state=d_mamba, expand=2, headdim=48)

        # 🔀 混合层（非线性特征压缩）
        self.mix = nn.Sequential(
            nn.Linear(d_model, mix_dim),
            nn.ReLU(),
            nn.Linear(mix_dim, mix_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 🎯 输出层（回归角度，单位：弧度）
        self.output_fc = nn.Linear(mix_dim, 1)
        nn.init.zeros_(self.output_fc.bias)
        nn.init.kaiming_uniform_(self.output_fc.weight, a=0.01)

    def forward(self, x, mic_pos):
        """
        x: 音频输入，形状 [B, T, C]，B=batch, T=时间长度, C=通道数
        mic_pos: 麦克风位置输入，形状 [B, 12]，每个样本4个mic * (x,y,z)
        """
        B, T, C = x.shape

        # 🎧 音频编码
        audio_feat = self.input_fc(x)         # (B, T, d_model)

        # 🧭 位置编码，扩展为序列形式 (B, T, d_model)
        mic_pos = mic_pos.view(B, -1)         # (B, 12)
        mic_feat = self.mic_fc(mic_pos)       # (B, d_model)
        mic_feat = mic_feat.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model)

        # 🔗 融合音频与位置特征
        x = audio_feat + mic_feat             # (B, T, d_model)

        # 🧠 Mamba主干 + 混合 + 平均池化
        x = self.mamba(x)
        x = self.mix(x)
        x = torch.mean(x, dim=1)              # (B, mix_dim)

        # 🎯 输出角度（单位：弧度）
        angle = self.output_fc(x)             # (B, 1)
        return angle.squeeze(1)               # (B,)
