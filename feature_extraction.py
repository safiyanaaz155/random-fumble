import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import os
import librosa
import torchaudio.transforms as T

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
# Emotions to observe
observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_mel_spectrogram(audio_path, sample_rate=22050, n_mels=40, n_fft=2048):
    try:
        file_name = os.path.basename(audio_path)
        emotion_code = file_name.split('-')[2]
        emotion_label = emotions.get(emotion_code, 'unknown')

        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono by taking the mean of stereo channels if necessary
        if waveform.shape[0] > 1:  
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft)
        mel_spectrogram = mel_transform(waveform)

        return mel_spectrogram

    except Exception as e:
        print(f"Error loading or processing audio file: {e}")
        return None

audio_path = r"C:\Users\rashe\OneDrive\Desktop\speech recognition\myenv\Actor_01\03-01-01-01-01-01-01.wav"
if os.path.isfile(audio_path):
    mel_spectrogram = extract_mel_spectrogram(audio_path)
    if mel_spectrogram is not None:
        print(f'Mel spectrogram shape: {mel_spectrogram.shape}')
    else:
        print("Failed to extract mel spectrogram.")
else:
    print(f"The specified path does not point to a file: {audio_path}")
