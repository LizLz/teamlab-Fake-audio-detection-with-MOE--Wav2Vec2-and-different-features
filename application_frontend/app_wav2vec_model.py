import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# --------------------------
# 1. Updated Model Class (with adaptive pooling)
# --------------------------
class SpoofDetectionModel(nn.Module):
    def __init__(self, hidden_dim, dropout_prob):
        super(SpoofDetectionModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(24)  # Fixed output size
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(128 * 24, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --------------------------
# 2. Feature Extraction
# --------------------------
def limit_audio_length(y, sr, target_length=4):
    target_samples = target_length * sr
    if len(y) >= target_samples:
        return y[:target_samples]
    else:
        repeats = target_samples // len(y) + 1
        return np.tile(y, repeats)[:target_samples]

def extract_wav2vec_features(waveform, sr):
    model_name = 'facebook/wav2vec2-base'
    pre_processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    inputs = pre_processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
        outputs = F.layer_norm(outputs, outputs.shape[-1:])
        wav2vec_tensor = outputs.permute(0, 2, 1)
    return wav2vec_tensor

# --------------------------
# 3. Spectrogram Saving
# --------------------------
def save_mel_spectrogram_image(mel_tensor, output_path):
    """
    Saves a Mel spectrogram tensor as a high-quality image for presentations or analysis,
    with axes, labels, and better resolution.
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)  # Higher DPI for clearer rendering
    im = ax.imshow(mel_tensor.numpy(), aspect='auto', origin='lower', cmap='magma')
    
    # Add axis labels and title
    ax.set_title('Log-Mel Spectrogram (dB)', fontsize=14)
    ax.set_xlabel('Time Frames', fontsize=12)
    ax.set_ylabel('Mel Frequency Bins', fontsize=12)
    
    # Color bar for intensity
    cbar = plt.colorbar(im, ax=ax, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)', fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

# --------------------------
# 4. Prediction Function
# --------------------------
def predict_with_pytorch(model, features):
    input_tensor = features.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs  # This is a tensor (logit)

# --------------------------
# 5. Model loading function
# --------------------------
@st.cache_resource
def load_model():
    hidden_dim = 256
    dropout_prob = 0.5
    model = SpoofDetectionModel(hidden_dim=hidden_dim, dropout_prob=dropout_prob)
    state_dict = torch.load("wav2vec_v2_updated_model_deepfake_audio_detection_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --------------------------
# 6. Streamlit App
# --------------------------
st.title("Deepfake Audio Detection (PyTorch, Wav2Vec2)")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"]
)
if uploaded_file is not None:
    os.makedirs("audio_files", exist_ok=True)
    audio_path = os.path.join("audio_files", uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load the waveform
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)
    sr = 16000

    st.audio(audio_path)  # Display the original audio (optional)

    st.write("### Spectrogram")
    # Compute Mel spectrogram and log scale
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform.squeeze())
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    spec_img_path = "melspectrogram.png"
    save_mel_spectrogram_image(mel_spec_db.cpu(), spec_img_path)
    st.image(spec_img_path)

    st.write("### Classification")
    features = extract_wav2vec_features(audio_path)
    st.write(f"Wav2Vec2 features shape: {features.shape}")  # Debug: should be [1, 768, T]
    model = load_model()
    logit = predict_with_pytorch(model, features)
    logit_value = logit.item()  # Convert tensor to float

    # Compute probability (after sigmoid)
    probability = torch.sigmoid(logit).item()

    # Display results (no thresholding, no class label)
    st.write(f"**Raw model output (logit):** {logit_value:.3f}")
    st.write(f"**Probability (bonafide):** {probability:.3f}")

else:
    st.info("Please upload an audio file")