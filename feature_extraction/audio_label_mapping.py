import torch
import os
import json
from tqdm import tqdm

# label the data

label_map = {'bonafide': 1, 'spoof':0}
audio_labels = {}

protocol_path = "D:/TeamLab/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

with open(protocol_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        file_name = parts[1]
        label = parts[-1]
        audio_labels[file_name] = label_map[label]

# preprocessing the data 

tensor_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/mel_segment_outputs/tensors_training_set" # the saved tensor dir
dataset = []

tensor_files = [f for f in os.listdir(tensor_dir) if f.endswith(".pt")]

for idx, file in enumerate(tqdm(tensor_files, desc="Processing data")):
    # load the tensor
    tensor_path = os.path.join(tensor_dir, file)
    log_mel_tensor = torch.load(tensor_path)

    file_name = os.path.splitext(file)[0]
    label = audio_labels[file_name]

    if label is not None:
        sample_data = {
            "id": idx,
            "file_name": file_name,
            "label": label,
            "features": log_mel_tensor.tolist()
        }
        dataset.append(sample_data)
    else:
        print(f"Warning: {file_name} not found in label mapping.")

# save dataset as a json file
output_dataset_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/mel_segment_outputs/labeled_dataset"
os.makedirs(output_dataset_dir, exist_ok=True) 
output_json_path = os.path.join(output_dataset_dir, "dataset_train.json")

# write the dataset to json file
with open(output_json_path, 'w') as f:
    for i, entry in enumerate(tqdm(dataset, desc="Dataset writing into JSON")):
        json.dump(entry, f)
        f.write('\n')  # Write each sample on a new line 

print(f"Dataset successfully saved to {output_json_path}")
print(dataset[0].shape)

    


