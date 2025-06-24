import os
import torch
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T

# Label mapping
label_map = {'bonafide': 1, 'spoof': 0}
audio_labels = {}

# === Paths ===
protocol_path = "D:\\Stuttgart\\(important) Third Semester\\Team lab\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.eval.trl.txt"
audio_base_path = "D:\\Stuttgart\\(important) Third Semester\\Team lab\\LA\\ASVspoof2019_LA_eval\\flac"
output_tensor_dir = "D:\\Stuttgart\\(important) Third Semester\\Team lab\\lfcc_eval"
os.makedirs(output_tensor_dir, exist_ok=True)

# === Load protocol labels ===
with open(protocol_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        file_name = parts[1]
        label = parts[-1]
        audio_labels[file_name] = label_map[label]

# === Utility: Limit audio length to 4 seconds ===
def limit_audio_length(waveform, sr, target_sec=4):
    target_samples = target_sec * sr
    current_samples = waveform.shape[1]
    if current_samples > target_samples:
        return waveform[:, :target_samples]
    else:
        repeats = (target_samples // current_samples) + 1
        extended = waveform.repeat(1, repeats)
        return extended[:, :target_samples]

# === LFCC extraction ===
def get_lfcc_tensor(audio_path):
    waveform, sr = torchaudio.load(audio_path, format="flac")
    waveform = limit_audio_length(waveform, sr)

    lfcc_transform = T.LFCC(
        sample_rate=sr,
        n_lfcc=40,
        speckwargs={"n_fft": 2048, "hop_length": 512}
    )
    lfcc = lfcc_transform(waveform)  # shape: [1, 40, timeframes]
    return lfcc.squeeze(0)           # shape: [40, timeframes]

# === Process audio files ===
audio_files = [f for f in os.listdir(audio_base_path) if f.endswith(".flac")]

for f in tqdm(audio_files, desc="Extracting and labeling"):
    file_path = os.path.join(audio_base_path, f)
    file_id = os.path.splitext(f)[0]
    if file_id not in audio_labels:
        print(file_id) 
        print(audio_base_path, f)
        # raise ValueError("File not found")
    #continue
    if file_id not in audio_labels:
        print(f"⚠️ Warning: No label found for {file_id}, skipping.")
        continue

    try:
        lfcc_tensor = get_lfcc_tensor(file_path)
        label = audio_labels[file_id]

        tensor_output_path = os.path.join(output_tensor_dir, f"{file_id}.pt")
        torch.save((lfcc_tensor, label), tensor_output_path)
    except Exception as e:
        print(f"❌ Error processing {file_id}: {e}")

print("✅ All audio processed and saved with labels.")