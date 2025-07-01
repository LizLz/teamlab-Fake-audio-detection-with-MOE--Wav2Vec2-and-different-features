import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# 1. LFCC + delta + delta delta model (static fc1 initialization)
# --------------------------
EXPECTED_TIMEFRAMES = 126  # Must match training

def get_fc1_input_dim():
    """
    Calculate the input dimension to fc1 after conv/pool stack.
    For input [120, 126]: after two MaxPool1d layers (kernel=2, stride=2): 126 -> 63 -> 31
    So output shape is [batch, 64, 31], so fc1_input_dim = 64*31 = 1984
    """
    pooled_frames = EXPECTED_TIMEFRAMES
    for _ in range(2):  # two pooling layers
        pooled_frames = pooled_frames // 2
    return 64 * pooled_frames

class SpoofDetectionModel(nn.Module):
    def __init__(self, hidden_dim, dropout_prob=0.5):
        super(SpoofDetectionModel, self).__init__()
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=120, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(dropout_prob)
        # Fully connected layers (static initialization)
        fc1_input_dim = get_fc1_input_dim()
        self.fc1 = nn.Linear(fc1_input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    def forward(self, x):
        # x: [batch, 120, timeframes]
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# --------------------------
# 2. Utility functions
# --------------------------

def limit_audio_length(waveform, sr, target_sec=4):
    """
    Crop or repeat-pad waveform to target_sec seconds.
    """
    target_samples = target_sec * sr
    current_samples = waveform.shape[1]
    if current_samples > target_samples:
        return waveform[:, :target_samples]
    else:
        repeats = (target_samples // current_samples) + 1
        extended = waveform.repeat(1, repeats)
        return extended[:, :target_samples]

def extract_lfcc_delta(audio_path):
    """
    Extract LFCC + delta + delta-delta features, pad/crop to EXPECTED_TIMEFRAMES.
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = limit_audio_length(waveform, sr)
    lfcc_transform = T.LFCC(
        sample_rate=sr,
        n_lfcc=40,
        speckwargs={"n_fft": 2048, "hop_length": 512}
    )
    lfcc = lfcc_transform(waveform).squeeze(0)  # [40, timeframes]
    delta = F.compute_deltas(lfcc)
    delta2 = F.compute_deltas(delta)
    features = torch.cat([lfcc, delta, delta2], dim=0)  # [120, timeframes]
    # Pad or crop to expected timeframes
    current_frames = features.shape[1]
    if current_frames < EXPECTED_TIMEFRAMES:
        pad_width = EXPECTED_TIMEFRAMES - current_frames
        features = torch.nn.functional.pad(features, (0, pad_width))
    elif current_frames > EXPECTED_TIMEFRAMES:
        features = features[:, :EXPECTED_TIMEFRAMES]
    return features

def save_spectrogram_image(audio_path, output_path='melspectrogram.png'):
    """
    Save a mel spectrogram image for visualization.
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze()
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(mel_spec_db.numpy(), aspect='auto', origin='lower')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return output_path

def predict_with_pytorch(model, features):
    """
    Run a forward pass and return predicted class and probability.
    """
    input_tensor = features.unsqueeze(0)  # [1, 120, timeframes]
    input_tensor = input_tensor.to(next(model.parameters()).device)
    with torch.no_grad():
        logit = model(input_tensor).item()
    pred = int(logit > 0)
    prob = torch.sigmoid(torch.tensor(logit)).item()
    return pred, prob

# --------------------------
# 3. Model loading function
# --------------------------
@st.cache_resource
def load_model():
    """
    Load the trained LFCC model and set to eval mode.
    """
    hidden_dim = 128
    dropout_prob = 0.5
    model = SpoofDetectionModel(hidden_dim=hidden_dim, dropout_prob=dropout_prob)
    state_dict = torch.load("lfcc_with_delta_v2_updated_model_deepfake_audio_detection_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --------------------------
# 4. Streamlit App
# --------------------------
st.title("Deepfake Audio Detection (PyTorch, LFCC + Delta + Delta-Delta)")

uploaded_file = st.file_uploader("Upload a WAV file", type="wav")
if uploaded_file:
    os.makedirs("audio_files", exist_ok=True)
    audio_path = os.path.join("audio_files", uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(audio_path)

    st.write("### Spectrogram")
    spec_img_path = save_spectrogram_image(audio_path)
    st.image(spec_img_path)

    st.write("### Classification")
    features = extract_lfcc_delta(audio_path)
    st.write(f"LFCC+delta features shape: {features.shape}")  # Debug: should be [120, timeframes]
    model = load_model()
    pred, prob = predict_with_pytorch(model, features)
    class_names = ["spoof", "bonafide"]
    st.write(f"**Prediction:** {class_names[pred]}")
    st.write(f"**Probability (bonafide):** {prob:.3f}")
else:
    st.info("Please upload a .wav file")
