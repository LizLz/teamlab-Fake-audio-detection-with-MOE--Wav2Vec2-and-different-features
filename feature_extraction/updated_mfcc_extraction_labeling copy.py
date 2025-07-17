import librosa
import numpy as np
import os
import torch
from tqdm import tqdm

# Label mapping
label_map = {'bonafide': 1, 'spoof': 0}
audio_labels = {}

# === Paths ===
protocol_path = "D:\\Felipe\\Team Lab\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.eval.trl.txt"
audio_base_path = "D:/Felipe/Team Lab/LA/ASVspoof2019_LA_eval/flac"
output_tensor_dir = "D:\\Felipe\\Team Lab\\teamlab-phonetics\\feature_extraction\\mfcc_segment_outputs\\mfcc_eval_set"
os.makedirs(output_tensor_dir, exist_ok=True)

# === Load protocol labels ===
with open(protocol_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        file_name = parts[1]
        label = parts[-1]
        audio_labels[file_name] = label_map[label]

# === Utility functions ===
def limit_audio_length(y, sr, target_length=4):
    target_samples = target_length * sr
    if len(y) > target_samples:
        return y[:target_samples]
    else:
        repeats = target_samples // len(y) + 1
        return np.tile(y, repeats)[:target_samples]

def get_mfcc_tensor(audio_path, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=None)
    y = limit_audio_length(y, sr)
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    except TypeError:
        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    # Optionally, you can normalize or take log(abs(mfcc)) if needed
    return torch.from_numpy(mfcc).float()

# === Process audio files ===
audio_files = [f for f in os.listdir(audio_base_path) if f.endswith(".flac")]

for f in tqdm(audio_files, desc="Extracting and labeling"):
    file_path = os.path.join(audio_base_path, f)
    file_id = os.path.splitext(f)[0]

    if file_id not in audio_labels:
        print(f"Warning: No label found for {file_id}, skipping.")
        continue

    mfcc_tensor = get_mfcc_tensor(file_path)
    label = audio_labels[file_id]

    tensor_output_path = os.path.join(output_tensor_dir, f"{file_id}.pt")
    torch.save((mfcc_tensor, label), tensor_output_path)

print("✅ All audio processed and saved with labels.")