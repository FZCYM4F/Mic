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
room_dim = [13.0, 13.0, 2.5]
rt60_tgt = 0.3
num_sources = 12
radii = [0.5, 1.0, 2.0, 3.0, 4.0]
yaw_angles = np.deg2rad([-60, -30, 0, 30, 60])  # ✅ 添加头部朝向（单位为弧度）

# ----------------------------------------
# 四个麦克风的空间坐标（原始定义）
# ----------------------------------------
mic_L = np.array([5.9147, 7.1864, 0.3314])
mic_R = np.array([5.9147, 4.8136, 0.3314])
mic_C = np.array([6.0000, 6.0000, 0.1941])
mic_T = np.array([3.4083, 6.0000, 0.0500])

head_center = mic_C.copy()
source_height = mic_C[2]
mic_labels = ['mic_L', 'mic_R', 'mic_C', 'mic_T']

# ✅ 添加：用于旋转头部三个麦克风（L/R/C）
def rotate_mics(mic_group, center, theta):
    # 只旋转 xy，z 轴保持不动
    rotated = []
    for mic in mic_group.T:
        offset = mic[:2] - center[:2]
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated_xy = R @ offset + center[:2]
        rotated.append([rotated_xy[0], rotated_xy[1], mic[2]])  # 保留原始 z
    return np.array(rotated).T

# ----------------------------------------
# 房间建模 & 吸收系数
# ----------------------------------------
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# ----------------------------------------
# 构建声源位置（多个半径 + 角度）
# ----------------------------------------
angles = np.linspace(0, 2 * np.pi, num_sources, endpoint=False)
source_positions = []
source_metadata = []

for r in radii:
    for a in angles:
        x = head_center[0] + r * np.cos(a)
        y = head_center[1] + r * np.sin(a)
        if 1.0 <= x <= room_dim[0] - 1.0 and 1.0 <= y <= room_dim[1] - 1.0:
            source_positions.append(np.array([x, y, source_height]))
            source_metadata.append({
                "radius": r,
                "angle_rad": a
            })
        else:
            print(f"⚠️ Skipped source at r={r:.1f}, θ={np.rad2deg(a):.1f}° (too close to wall)")

# ----------------------------------------
# 加载语音数据（例：说话人 32 文件夹 4137）
# ----------------------------------------
target_path = "/scratch/ym4c23/Mic/dataset/Librispeech/train-clean-100-wav/32/21625"
all_audio_files = glob.glob(os.path.join(target_path, "*.wav"))
print(f"✅ 加载音频 {len(all_audio_files)} 条")

# ----------------------------------------
# 输出路径设置
# ----------------------------------------
output_root = "/scratch/ym4c23/Mic/Step3_/result"
multi_wav_dir = os.path.join(output_root, "evalminiaudio")
single_mic_dir = os.path.join(multi_wav_dir, "evalminisingle")
npy_dir = os.path.join(output_root, "evalmininp")
meta_path = os.path.join(output_root, "evalminiCSV/angles_with_mics.csv")

os.makedirs(multi_wav_dir, exist_ok=True)
os.makedirs(single_mic_dir, exist_ok=True)
os.makedirs(npy_dir, exist_ok=True)
os.makedirs(os.path.dirname(meta_path), exist_ok=True)

# ----------------------------------------
# 主循环：每个源位置 × 每个 yaw 角度
# ----------------------------------------
metadata = []
sample_index = 0

for audio_path in all_audio_files:
    fs_audio, audio = wavfile.read(audio_path)
    audio = audio.astype(np.float32)

    for i, src_pos in enumerate(source_positions):
        for yaw in yaw_angles:

            # ✅ 旋转头部的 L/R/C 麦克风（T 保持不动）
            rotated_mics = rotate_mics(np.c_[mic_L, mic_R, mic_C], head_center, yaw)
            rotated_mic_L, rotated_mic_R, rotated_mic_C = rotated_mics.T
            rotated_mic_T = mic_T

            mic_positions = np.c_[
                rotated_mic_L,
                rotated_mic_R,
                rotated_mic_C,
                rotated_mic_T
            ]

            # 创建房间并添加信源与麦克风
            room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption),
                               max_order=max_order, air_absorption=True)
            room.add_source(src_pos, signal=audio, delay=0.5)
            room.add_microphone_array(mic_positions)
            room.simulate()

            # 保存多通道和单通道音频
            multi_wav_path = os.path.join(multi_wav_dir, f"source_{sample_index:05d}_multi.wav")
            room.mic_array.to_wav(multi_wav_path, norm=True, bitdepth=np.int16)
            np.save(os.path.join(npy_dir, f"source_{sample_index:05d}.npy"), room.mic_array.signals)

            for m in range(4):
                mono_path = os.path.join(single_mic_dir, f"source_{sample_index:05d}_{mic_labels[m]}.wav")
                signal = room.mic_array.signals[m]
                max_val = np.max(np.abs(signal))
                signal = signal / max_val * 32767 if max_val > 1e-8 else signal * 32767
                wavfile.write(mono_path, fs, signal.astype(np.int16))

            # ✅ 计算相对当前头部方向的角度（弧度）
            delta_head = src_pos[:2] - head_center[:2]
            angle_relative_to_head = np.arctan2(delta_head[1], delta_head[0]) - (np.pi / 2 + yaw)
            angle_relative_to_head = np.arctan2(np.sin(angle_relative_to_head), np.cos(angle_relative_to_head))

            mic_angles = []
            for mic_pos in [rotated_mic_L, rotated_mic_R, rotated_mic_C, rotated_mic_T]:
                delta = src_pos[:2] - mic_pos[:2]
                angle = np.arctan2(delta[1], delta[0]) - (np.pi / 2 + yaw)
                angle = np.arctan2(np.sin(angle), np.cos(angle))
                mic_angles.append(angle)

            # ✅ 保存 metadata
            metadata.append({
                "source_index": sample_index,
                "source_wav": os.path.basename(audio_path),
                "source_x": src_pos[0],
                "source_y": src_pos[1],
                "source_z": src_pos[2],
                "radius": source_metadata[i]["radius"],
                "angle_rad": source_metadata[i]["angle_rad"],
                "yaw_deg": np.rad2deg(yaw),
                "theta": yaw,
                "angle_relative_to_head": angle_relative_to_head,
                # ✅ 所有麦克风的位置（补全）
                "mic_L_x": rotated_mic_L[0], "mic_L_y": rotated_mic_L[1], "mic_L_z": rotated_mic_L[2],
                "mic_R_x": rotated_mic_R[0], "mic_R_y": rotated_mic_R[1], "mic_R_z": rotated_mic_R[2],
                "mic_C_x": rotated_mic_C[0], "mic_C_y": rotated_mic_C[1], "mic_C_z": rotated_mic_C[2],
                "mic_T_x": rotated_mic_T[0], "mic_T_y": rotated_mic_T[1], "mic_T_z": rotated_mic_T[2],
                "angle_mic_L": mic_angles[0],
                "angle_mic_R": mic_angles[1],
                "angle_mic_C": mic_angles[2],
                "angle_mic_T": mic_angles[3],
            })

            sample_index += 1
            del room

# ----------------------------------------
# 保存 CSV metadata
# ----------------------------------------
df = pd.DataFrame(metadata)
df.to_csv(meta_path, index=False)
print(f"✅ 生成完毕，共 {sample_index} 条样本。")
