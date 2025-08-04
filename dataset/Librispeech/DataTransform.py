import os
from pathlib import Path
import torchaudio

# 设置路径
input_dir = Path("/home/ym4c23/Mic/dataset/Librispeech/train-clean-100")  # 修改为你本地路径
output_dir = Path("/home/ym4c23/Mic/dataset/Librispeech/train-clean-100-wav")         # 目标路径
output_dir.mkdir(parents=True, exist_ok=True)

# 遍历所有 .flac 文件
flac_files = list(input_dir.rglob("*.flac"))

for flac_path in flac_files:
    rel_path = flac_path.relative_to(input_dir).with_suffix(".wav")
    wav_path = output_dir / rel_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    waveform, sr = torchaudio.load(flac_path)
    torchaudio.save(str(wav_path), waveform, sr)

    print(f"✅ Converted: {flac_path} -> {wav_path}")
