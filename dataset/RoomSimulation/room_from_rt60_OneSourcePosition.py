import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# The desired reverberation time and dimensions of the room
rt60 = 0.3  # seconds
room_dim = [4.0, 4.0, 2.5]  # meters

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

# Create the room
room = pra.ShoeBox(
    room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
)

# import a mono wavfile as the source signal
fs, audio = wavfile.read("/home/ym4c23/Mic/dataset/Librispeech/train-clean-100-wav/19/198/19-198-0000.wav")
# Add a Sound Source
room.add_source([3.0, 2.0, 1.5], signal=audio, delay=0.2)


robot_center = np.array([2.0, 2.0, 0.1651])
# define the locations of the microphones
mic_L = robot_center + np.array([-0.0474,  0.0519, 0.0827])  # left ear
mic_R = robot_center + np.array([-0.0474, -0.0519, 0.0827])  # right ear
mic_C = robot_center + np.array([ 0.0500,  0.0400, 0.0250])  # body
mic_T = robot_center + np.array([-0.1600, 0.0000, 0.0400])   # tail
mic_locs = np.c_[
    mic_L,
    mic_R,
    mic_C,
    mic_T
]

# finally place the array in the room
room.add_microphone_array(mic_locs)

 # Run the simulation (this will also build the RIR automatically)
room.simulate()

room.mic_array.to_wav(
    f"/home/ym4c23/Mic/dataset/RoomSimulation/result_OneSourcePosition/example_speech_reverb.wav",
    norm=True,
    bitdepth=np.int16,
)

 # measure the reverberation time
rt60 = room.measure_rt60()
print("The desired RT60 was {}".format(rt60))
print("The measured RT60 is {}".format(rt60[1, 0]))

# plot the RIRs
select = None  # plot all RIR
# select = (2, 0)  # uncomment to only plot the RIR from mic 2 -> src 0
# select = [(0, 0), (2, 0)]  # only mic 0 -> src 0, mic 2 -> src 0
fig, axes = room.plot_rir(select=select, kind="ir")  # impulse responses
fig, axes = room.plot_rir(select=select, kind="tf")  # transfer function
fig, axes = room.plot_rir(select=select, kind="spec")  # spectrograms

plt.tight_layout()
plt.show()