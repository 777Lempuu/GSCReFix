import streamlit as st
import torch
import torchaudio
from torchaudio.transforms import Resample, MFCC
import numpy as np

# Configuration (match your training setup)
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 400
HOP_LENGTH = 160
LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']  # Update with your actual labels

@st.cache_resource
def load_model():
    # Initialize a simple model structure (must match your training architecture)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, len(LABELS))
    
    # Load your trained weights
    state_dict = torch.load('GSC_ReFix.pt', map_location='cpu')['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_audio(waveform, sample_rate):
    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        resampler = Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Pad/trim to 1 second
    if waveform.shape[1] < SAMPLE_RATE:
        waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.shape[1]))
    else:
        waveform = waveform[:, :SAMPLE_RATE]
    
    # Extract MFCC features
    mfcc_transform = MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={'n_fft': N_FFT, 'hop_length': HOP_LENGTH, 'n_mels': 80, 'center': False}
    )
    return mfcc_transform(waveform)

# Streamlit UI
st.title("Speech Command Classifier")
st.write("Upload a 1-second audio clip with a spoken command")

uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file:
    try:
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(uploaded_file)
        features = preprocess_audio(waveform, sample_rate)
        
        # Load model and predict
        model = load_model()
        with torch.no_grad():
            logits = model(features.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
        
        # Display results
        st.success(f"Predicted: {LABELS[top_idx]} ({(top_prob.item()*100):.1f}% confidence)")
        
        # Show all probabilities
        st.write("All predictions:")
        for i, prob in enumerate(probs.squeeze().numpy()):
            st.progress(float(prob), text=f"{LABELS[i]}: {prob:.1%}")
            
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
