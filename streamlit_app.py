import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch import nn
import gdown
import os

# Model definition (must match your training architecture)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AudioResNet(nn.Module):
    def __init__(self, num_classes=35):
        super(AudioResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Audio preprocessing
class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = 16000
        self.n_mfcc = 40
        self.n_fft = 400
        self.hop_length = 160
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': 80,
                'center': False
            }
        )
        
    def __call__(self, waveform, sample_rate):
        if sample_rate != self.sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if waveform.shape[1] < self.sample_rate:
            waveform = F.pad(waveform, (0, self.sample_rate - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.sample_rate]
            
        mfcc = self.mfcc_transform(waveform)
        if torch.isnan(mfcc).any() or torch.isinf(mfcc).any():
            mfcc = torch.nan_to_num(mfcc, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return mfcc

# Labels from Google Speech Commands v2
labels = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

# Download model from Google Drive
@st.cache_resource
def load_model():
    model_path = 'GSC_ReFix.pt'
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/uc?id=1XcCw-c71St-895szf861FVuKrP9YQ0zA'
        gdown.download(url, model_path, quiet=False)
    
    model = AudioResNet(num_classes=len(labels))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    return model

# Streamlit app
def main():
    st.title("Speech Command Classifier")
    st.write("Upload an audio file or record your voice to classify speech commands")
    
    model = load_model()
    preprocessor = AudioPreprocessor()
    
    audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])
    
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        
        if st.button("Classify"):
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_file)
            features = preprocessor(waveform, sample_rate)
            features = features.unsqueeze(0)  # Add batch dimension
            
            # Predict
            with torch.no_grad():
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                predicted_label = labels[predicted.item()]
                
            st.success(f"Predicted command: **{predicted_label}**")

if __name__ == "__main__":
    main()
