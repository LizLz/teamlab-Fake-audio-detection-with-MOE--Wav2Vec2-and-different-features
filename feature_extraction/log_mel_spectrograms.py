import librosa
import numpy as np
import os
import torch
import random
from tqdm import tqdm

# Set a fixed random seed for reproducibility
#random.seed(42)  # The number can be any integer, but I've used 42 as an arbitrary choice to keep it consistent.

# Paths to the audio files and output directories
#protocol_file = 'D:/TeamLab/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
#audio_base_path = 'D:/TeamLab/dataset/LA/ASVspoof2019_LA_train/flac'

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
        # Cut the audio to the target length
        y = y[:target_samples]
    else:
        # Loop the audio until it reaches the target length
        num_repeats = target_samples // len(y) + 1
        y = np.tile(y, num_repeats)[:target_samples]
    return y

def get_Melspec(audio_file):
    """
    Extracts the log Mel spectrogram from an audio file and converts it to a PyTorch tensor.

    Parameters:
        audio_file (str): Path to the audio file in .flac format.

    Returns:
        torch.Tensor: Log Mel spectrogram as a PyTorch tensor.
    """
    y, sr = librosa.load(audio_file, sr=None)
    y = limit_audio_length(y, sr)  # Limit the audio to 4 seconds
    try:
        # Attempt to use the modern signature of melspectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    except TypeError:
        # Fallback for older versions of librosa
        mel_spectrogram = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    log_mel_tensor = torch.from_numpy(log_mel_spectrogram).float()
    return log_mel_tensor

#def get_audio_files(folder_path, num_files):    
#    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.flac')]
#    selected_files = random.sample(all_files, min(num_files, len(all_files)))
#    return selected_files

def save_mel_spectrogram_image(mel_tensor, output_path):
    """
    Saves a Mel spectrogram tensor as an image.

    Parameters:
        mel_tensor (torch.Tensor): The Mel spectrogram tensor.
        output_path (str): Path to save the image.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_tensor.numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Mel Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency Bins')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

### Main script to process audio files and save mel spectrograms ###
folder_path = "D:/TeamLab/dataset/LA/ASVspoof2019_LA_dev/flac"
#num_files_to_process = 3000  # Change this to the desired number of files

# Update selected_audio_files to process all .flac files in the folder
selected_audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.flac')]

mel_tensors = []
for audio_file in tqdm(selected_audio_files, desc="Processing audio files"): #uses tqdm to show progress bar
    # Load and process each audio file
    mel_tensor = get_Melspec(audio_file)
    mel_tensors.append(mel_tensor)

### Save tensors and images ###
output_tensor_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/mel_segment_outputs/tensors_development_set"
os.makedirs(output_tensor_dir, exist_ok=True) # Create the output directory if it doesn't exist

## Save tensors ##
# Update the tensor saving logic to retain the original file names
for audio_file, mel_tensor in zip(selected_audio_files, mel_tensors):
    original_name = os.path.splitext(os.path.basename(audio_file))[0]  # Extract the original file name without extension
    tensor_path = os.path.join(output_tensor_dir, f'{original_name}.pt')  # Use the original name for the tensor file
    torch.save(mel_tensor, tensor_path)

# Uncomment the following lines to save the images as well ##
#output_image_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/mel_segment_outputs/images_development_set"
#os.makedirs(output_image_dir, exist_ok=True)
#for i, mel_tensor in enumerate(tqdm(mel_tensors, desc="Saving images")):
#    image_path = os.path.join(output_image_dir, f'mel_segment_{i:03d}.png')
#    save_mel_spectrogram_image(mel_tensor, image_path)

print("Mel spectrograms, tensors, and images (if selected) saved successfully.")
print(mel_tensors[0].shape)  # Print the shape of the first tensor for verification
print(mel_tensors[1].shape)  # Print the shape of the second tensor for verification