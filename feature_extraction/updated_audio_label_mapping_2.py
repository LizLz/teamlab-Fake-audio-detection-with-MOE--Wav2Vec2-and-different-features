import torch
import os
from tqdm import tqdm

# Label mapping
label_map = {'bonafide': 1, 'spoof': 0}
audio_labels = {}

# Path to the protocol file
protocol_path = "D:/TeamLab/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# Read labels into dictionary
with open(protocol_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        file_name = parts[1]
        label = parts[-1]
        audio_labels[file_name] = label_map[label]

# Directory where tensors are stored
tensor_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/mel_segment_outputs/tensors_training_set"

# List all .pt files
tensor_files = [f for f in os.listdir(tensor_dir) if f.endswith(".pt")]

# Process and overwrite each file with (tensor, label)
for file in tqdm(tensor_files, desc="Saving tensors with labels"):
    tensor_path = os.path.join(tensor_dir, file)
    log_mel_tensor = torch.load(tensor_path)

    file_name = os.path.splitext(file)[0]
    label = audio_labels.get(file_name)

    if label is not None:
        torch.save((log_mel_tensor, label), tensor_path)  # Overwrite
    else:
        print(f"Warning: {file_name} not found in label mapping.")




