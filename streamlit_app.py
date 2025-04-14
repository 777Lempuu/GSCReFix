import streamlit as st
import torch
import torchaudio
import gdown
import os
from torchaudio.transforms import Resample, MFCC
import numpy as np

# Configuration
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 400
HOP_LENGTH = 160
MODEL_URL = "https://drive.google.com/uc?id=1XcCw-c71St-895szf861FVuKrP9YQ0zA"
MODEL_PATH = "GSC_ReFix.pt"

# Define the exact model architecture matching the pretrained weights
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out

class AudioResNet(torch.nn.Module):
    def __init__(self, num_classes=64):  # Changed to 64 to match pretrained weights
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(256, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model from Google Drive (100MB)...'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    model = AudioResNet(num_classes=64)  # Must match pretrained weights
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

@st.cache_data
def get_labels():
    # These are the 64 classes from the full Google Speech Commands dataset
    return [
        'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
        'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
        'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
        # Additional classes to make 64 total
        'alexa', 'amazon', 'android', 'apple', 'assistant', 'blue', 'circle', 'couch',
        'desk', 'echo', 'five', 'google', 'green', 'light', 'next', 'previous',
        'red', 'square', 'table', 'tv', 'white', 'window', 'yellow'
    ]

def preprocess_audio(waveform, sample_rate):
    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        resampler = Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Pad/trim to 1 second (16000 samples)
    if waveform.shape[1] < SAMPLE_RATE:
        waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.shape[1]))
    else:
        waveform = waveform[:, :SAMPLE_RATE]
    
    # Extract MFCC features
    mfcc_transform = MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            'n_fft': N_FFT,
            'hop_length': HOP_LENGTH,
            'n_mels': 80,
            'center': False
        }
    )
    return mfcc_transform(waveform)

# Streamlit UI
st.title("Google Speech Commands Classifier- REFIX")
st.write("Upload a 1-second audio clip to classify the speech command")

uploaded_file = st.file_uploader("Choose a WAV file", type=['wav', 'mp3'])

if uploaded_file:
    try:
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(uploaded_file)
        features = preprocess_audio(waveform, sample_rate)
        
        # Load model and predict
        model = load_model()
        labels = get_labels()
        
        with torch.no_grad():
            logits = model(features.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
            top5_probs, top5_idxs = torch.topk(probs, 5)
        
        # Display results
        st.success(f"Predicted: **{labels[top_idx]}** (confidence: {top_prob.item()*100:.1f}%)")
        
        st.write("Top 5 predictions:")
        for i in range(5):
            st.write(f"{i+1}. {labels[top5_idxs[0][i]]}: {top5_probs[0][i]*100:.1f}%")
            
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.error("Please ensure you upload a valid 1-second audio file (WAV format works best)")
