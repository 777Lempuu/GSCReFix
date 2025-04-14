import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T
import gdown
import os
import torch.nn.functional as F

# Simplified Model Architecture
class SpeechCommandModel(torch.nn.Module):
    def __init__(self, num_classes=35):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Audio Preprocessor
class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = 16000
        self.mfcc = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=40,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 80}
        )

    def __call__(self, waveform, sample_rate):
        if sample_rate != self.sample_rate:
            waveform = T.Resample(sample_rate, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] < self.sample_rate:
            waveform = F.pad(waveform, (0, self.sample_rate - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.sample_rate]
        return self.mfcc(waveform)

# Labels
LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

# Model Loader
@st.cache_resource
def load_model():
    model_path = 'model.pt'
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/uc?id=1XcCw-c71St-895szf861FVuKrP9YQ0zA'
        gdown.download(url, model_path, quiet=True)
    
    model = SpeechCommandModel(num_classes=len(LABELS))
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Streamlit App
def main():
    st.title("Speech Command Classifier")
    st.write("Upload a 1-second audio clip (WAV format recommended)")
    
    model = load_model()
    preprocessor = AudioPreprocessor()
    
    audio_file = st.file_uploader("Choose file", type=['wav', 'mp3'])
    
    if audio_file:
        st.audio(audio_file)
        if st.button("Classify"):
            try:
                waveform, sample_rate = torchaudio.load(audio_file)
                features = preprocessor(waveform, sample_rate).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(features)
                    prediction = LABELS[outputs.argmax().item()]
                
                st.success(f"Predicted: {prediction}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
