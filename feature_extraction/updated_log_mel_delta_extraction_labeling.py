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
output_tensor_dir = "D:\\Felipe\\Team Lab\\teamlab-phonetics\\feature_extraction\\logmel_segment_outputs\\logmel_delta_eval_set"
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

def get_log_mel_tensor_with_deltas(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y = limit_audio_length(y, sr)
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    except TypeError:
        mel = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
    logmel_full = np.concatenate([log_mel, delta, delta2], axis=0)
    return torch.from_numpy(logmel_full).float()

# === Process audio files ===
audio_files = [f for f in os.listdir(audio_base_path) if f.endswith(".flac")]

for f in tqdm(audio_files, desc="Extracting and labeling"):
    file_path = os.path.join(audio_base_path, f)
    file_id = os.path.splitext(f)[0]

    if file_id not in audio_labels:
        print(f"Warning: No label found for {file_id}, skipping.")
        continue

    log_mel_tensor = get_log_mel_tensor_with_deltas(file_path)
    label = audio_labels[file_id]

    tensor_output_path = os.path.join(output_tensor_dir, f"{file_id}.pt")
    torch.save((log_mel_tensor, label), tensor_output_path)

print("✅ All audio processed and saved with labels.")