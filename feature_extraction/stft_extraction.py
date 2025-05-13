import librosa
import numpy as np
import os
import torch
import random
from tqdm import tqdm

# Set a fixed random seed for reproducibility
random.seed(42)

def limit_audio_length(y, sr, target_length=4):
    """
    Limits the audio to a target length in seconds. If the audio is shorter, it loops until the target length.

    Parameters:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.
        target_length (int): Target length in seconds (default is 4 seconds).

    Returns:
        np.ndarray: Audio time series limited to the target length.
    """
    target_samples = target_length * sr
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        num_repeats = target_samples // len(y) + 1
        y = np.tile(y, num_repeats)[:target_samples]
    return y

def get_STFT(audio_file):
    """
    Extracts the STFT features from an audio file and converts it to a PyTorch tensor.

    Parameters:
        audio_file (str): Path to the audio file in .flac format.

    Returns:
        torch.Tensor: STFT features as a PyTorch tensor.
    """
    y, sr = librosa.load(audio_file, sr=None)
    y = limit_audio_length(y, sr)  # Limit the audio to 4 seconds
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    stft_magnitude = np.abs(stft)
    stft_tensor = torch.from_numpy(stft_magnitude).float()
    return stft_tensor

def save_stft_spectrogram_image(stft_tensor, output_path):
    """
    Saves a STFT spectrogram tensor as an image.

    Parameters:
        STFT_tensor (torch.Tensor): The STFT spectrogram tensor.
        output_path (str): Path to save the image.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.imshow(stft_tensor.numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('STFT Frequency Bins')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

folder_path = "D:/TeamLab/dataset/LA/ASVspoof2019_LA_train/flac"
num_files_to_process = 3000
selected_audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.flac')][:num_files_to_process]

stft_tensors = []
for audio_file in tqdm(selected_audio_files, desc="Processing audio files"):
    stft_tensor = get_STFT(audio_file)
    stft_tensors.append(stft_tensor)

output_tensor_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/stft_segment_outputs/tensors_training_set"
os.makedirs(output_tensor_dir, exist_ok=True)

for i, stft_tensor in enumerate(tqdm(stft_tensors, desc="Saving tensors")):
    tensor_path = os.path.join(output_tensor_dir, f'stft_segment_{i:03d}.pt')
    torch.save(stft_tensor, tensor_path)

# Uncomment the following lines to save the images as well ##
output_image_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/stft_segment_outputs/images_training_set"
os.makedirs(output_image_dir, exist_ok=True)
for i, stft_tensor in enumerate(tqdm(stft_tensors, desc="Saving images")):
    image_path = os.path.join(output_image_dir, f'mel_segment_{i:03d}.png')
    save_stft_spectrogram_image(stft_tensor, image_path)

print("STFT tensors saved successfully.")
print(stft_tensors[0].shape)  # Print the shape of the first tensor for verification
print(stft_tensors[1].shape)  # Print the shape of the second tensor for verificat