import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox
import torchaudio.transforms as T

# Model class definition 
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 250, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Emotion dictionary and observed emotions
emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Function to load the trained model
def load_model():
    model = EmotionRecognitionModel(num_classes=len(observed_emotions))
    model.load_state_dict(torch.load("C:\\Users\\rashe\\OneDrive\\Desktop\\speech recognition\\my_trained_model.pth"))
    model.eval()
    return model

# Function to extract mel spectrogram
def extract_mel_spectrogram(audio_path, sample_rate=22050, n_mels=64):
    waveform, _ = torchaudio.load(audio_path)
    
    # Convert to mono 
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    mel_spectrogram = mel_transform(waveform)
    fixed_length = 1000
    n_frames = mel_spectrogram.shape[-1]
    if n_frames > fixed_length:
        mel_spectrogram = mel_spectrogram[:, :, :fixed_length]
    elif n_frames < fixed_length:
        padding = fixed_length - n_frames
        pad_tensor = torch.zeros((mel_spectrogram.shape[0], mel_spectrogram.shape[1], padding))
        mel_spectrogram = torch.cat((mel_spectrogram, pad_tensor), dim=-1)
    
    mel_spectrogram = mel_spectrogram.unsqueeze(0)  
    
    return mel_spectrogram

# predict emotion
def predict_emotion(model, audio_path):
    spectrogram = extract_mel_spectrogram(audio_path)
    with torch.no_grad():
        output = model(spectrogram)
        _, predicted = torch.max(output, 1)
        return observed_emotions[predicted.item()]

# open file dialog and select audio file
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        result.set(f"Selected file: {file_path}")
        return file_path
    else:
        messagebox.showinfo("File Selection", "No file selected!")
        return None

# handle the prediction button click
def on_predict():
    file_path = select_file()
    if file_path:
        try:
            emotion = predict_emotion(model, file_path)
            result.set(f"Predicted Emotion: {emotion}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict emotion: {e}")

# Initialize the model
model = load_model()
root = tk.Tk()
root.title("Speech Emotion Recognition")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

result = tk.StringVar()
result.set("Select an audio file and predict emotion")

label = tk.Label(frame, textvariable=result)
label.pack()

predict_button = tk.Button(frame, text="Select Audio and Predict Emotion", command=on_predict)
predict_button.pack(pady=10)

root.mainloop()
