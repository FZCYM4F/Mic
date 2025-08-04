import os
import glob
import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra
import pandas as pd

# ----------------------------------------
# 模拟参数
# ----------------------------------------
fs = 16000
room_dim = [6.0, 6.0, 2.5]
rt60_tgt = 0.3
radius = 1.5
num_sources = 12

# ----------------------------------------
# 四个麦克风的空间坐标
# ----------------------------------------
mic_L = np.array([3.1053, 3.1864, 0.3314])  # left ear
mic_R = np.array([3.1053, 0.8136, 0.3314])  # right ear
mic_C = np.array([3.1906, 2.0000, 0.1941])  # head center
mic_T = np.array([0.5989, 2.0000, 0.0500])  # tail

head_center = mic_C.copy()
source_height = mic_C[2]  # the height of source and microphones are identical

mic_positions = np.c_[
    mic_L,
    mic_R,
    mic_C,
    mic_T
]
mic_labels = ['mic_L', 'mic_R', 'mic_C', 'mic_T']

# ----------------------------------------
# 房间建模 & 吸收系数
# ----------------------------------------
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# ----------------------------------------
# 构建声源位置：绕头部一圈，等角度分布
# ----------------------------------------
angles = np.linspace(0, 2 * np.pi, num_sources, endpoint=False)
source_positions = [
    np.array([
        head_center[0] + radius * np.cos(a),
        head_center[1] + radius * np.sin(a),
        source_height
    ])
    for a in angles
]

# ----------------------------------------
# 读取前 10 位说话人下的所有 .wav 文件
# ----------------------------------------
librispeech_root = "/mainfs/scratch/ym4c23/Mic/dataset/Librispeech/train-clean-100-wav/"
speaker_dirs = [sorted(os.listdir(librispeech_root))[22]]  # 选前 10 个说话人
all_audio_files = []
for speaker in speaker_dirs:
    speaker_path = os.path.join(librispeech_root, speaker)
    if os.path.isdir(speaker_path):
        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)
            if os.path.isdir(chapter_path):
                all_audio_files += glob.glob(os.path.join(chapter_path, "*.wav"))

# ----------------------------------------
# 数据输出路径（你已经提前创建好了）
# ----------------------------------------
output_root = "/mainfs/scratch/ym4c23/Mic/dataset/RoomSimulation/result_OneSourceMoving/"
multi_wav_dir = os.path.join(output_root, "evalaudio")
single_mic_dir = os.path.join(multi_wav_dir, "evalsingle")
npy_dir = os.path.join(output_root, "evalnp")
meta_path = os.path.join(output_root, "evalCSV/angles_with_mics.csv")

# ----------------------------------------
# 主循环：遍历语音文件并生成多角度模拟
# ----------------------------------------
metadata = []
sample_index = 0

for audio_path in all_audio_files:
    fs_audio, audio = wavfile.read(audio_path)
    audio = audio.astype(np.float32)

    for i, src_pos in enumerate(source_positions):
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            air_absorption=True,
        )
        room.add_source(src_pos, signal=audio, delay=0.5)
        room.add_microphone_array(mic_positions)
        room.simulate()

        # save 4-channel audio
        multi_wav_path = os.path.join(multi_wav_dir, f"source_{sample_index:04d}_multi.wav")
        room.mic_array.to_wav(multi_wav_path, norm=True, bitdepth=np.int16)

        np.save(os.path.join(npy_dir, f"source_{sample_index:04d}.npy"), room.mic_array.signals)

        # save audio from each mic
        for m in range(4):
            mono_path = os.path.join(single_mic_dir, f"source_{sample_index:04d}_{mic_labels[m]}.wav")
            signal = room.mic_array.signals[m]
            max_val = np.max(np.abs(signal))
            if max_val > 1e-8:
                signal = signal / max_val * 32767
            else:
                print(f"⚠️ Mic {mic_labels[m]} signal too small, silence likely.")
                signal = signal * 32767
            wavfile.write(mono_path, fs, signal.astype(np.int16))

        # calculate angle relative to robot head center (robot facing +Y)
        delta_head = src_pos[:2] - head_center[:2]
        angle_head = np.arctan2(delta_head[1], delta_head[0]) - np.pi / 2
        angle_head = np.arctan2(np.sin(angle_head), np.cos(angle_head))

        # Generalized head orientation: use this to simulate head turning
        head_orientation_rad = 0.0  # 例如 np.deg2rad(-30) 表示头部向左偏转 30 度

        # angle relative to head direction
        angle_head = angle_head - head_orientation_rad
        angle_head = np.arctan2(np.sin(angle_head), np.cos(angle_head))  # normalize to [-π, π]

        # calculate angle for each mic
        mic_angles = []
        for mic_pos in [mic_L, mic_R, mic_C, mic_T]:
            delta = src_pos[:2] - mic_pos[:2]
            angle = np.arctan2(delta[1], delta[0]) - np.pi / 2
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            mic_angles.append(angle)

        # store data
        metadata.append({
            "source_index": sample_index,
            "source_wav": os.path.basename(audio_path),
            "source_x": src_pos[0],
            "source_y": src_pos[1],
            "source_z": src_pos[2],
            "angle_center": angle_head,
            "angle_head": angle_head,
            "mic_L_x": mic_L[0],
            "mic_L_y": mic_L[1],
            "mic_L_z": mic_L[2],
            "angle_mic_L": mic_angles[0],
            "angle_mic_R": mic_angles[1],
            "angle_mic_C": mic_angles[2],
            "angle_mic_T": mic_angles[3],
        })

        sample_index += 1

# ----------------------------------------
# 保存元数据 CSV
# ----------------------------------------
df = pd.DataFrame(metadata)
df.to_csv(meta_path, index=False)

print(f"✅ 模拟完成，处理了 {sample_index} 条样本，来自 {len(all_audio_files)} 条语音。")
