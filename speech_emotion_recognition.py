import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

# Emotion dictionary and observed emotions
emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None, fixed_length=1000):
        super(EmotionDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.fixed_length = fixed_length
        self.data = [os.path.join(root, file) for root, _, files in os.walk(root_dir) for file in files if file.endswith('.wav')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]
        mel_spectrogram, label = self.extract_mel_spectrogram(audio_path)

        if mel_spectrogram is not None and label in observed_emotions:
            mel_spectrogram = self.pad_or_truncate_spectrogram(mel_spectrogram)
            if self.transform:
                mel_spectrogram = self.transform(mel_spectrogram)
                
            label_index = observed_emotions.index(label)
            label_tensor = torch.tensor(label_index, dtype=torch.long) 
            
            return mel_spectrogram, label_tensor
        
        else:
            # Skip invalid data
            return None, None

    def extract_mel_spectrogram(self, audio_path, sample_rate=22050, n_mels=64):
        try:
            emotion_code = os.path.basename(audio_path).split('-')[2]
            emotion_label = emotions.get(emotion_code, 'unknown')

            waveform, _ = torchaudio.load(audio_path)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
            mel_spectrogram = mel_transform(waveform)

            return mel_spectrogram, emotion_label
    
        except Exception as e:
            print(f"Error loading or processing audio file: {e}")
            return None, None

    def pad_or_truncate_spectrogram(self, spectrogram):
        n_frames = spectrogram.shape[-1]
        if n_frames > self.fixed_length:
            # Truncate
            spectrogram = spectrogram[:, :, :self.fixed_length]
        elif n_frames < self.fixed_length:
            # Pad
            padding = self.fixed_length - n_frames
            pad_tensor = torch.zeros((spectrogram.shape[0], spectrogram.shape[1], padding))
            spectrogram = torch.cat((spectrogram, pad_tensor), dim=-1)
        
        return spectrogram

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjust the dimensions based on dummy pass results
        self.fc1 = nn.Linear(64 * 16 * 250, 128)  # Use the correct size from dummy pass
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        print("Starting forward pass")
        x = self.pool(F.relu(self.conv1(x)))
        print("After conv1 and pooling:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print("After conv2 and pooling:", x.shape)
        x = x.view(x.size(0), -1)
        print("Shape before flattening:", x.shape)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, device, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i, (spectrogram, label) in enumerate(train_loader):
            if spectrogram is not None and label is not None:
                print(f"Epoch {epoch+1}, Batch {i+1}")
                print(f"Spectrogram shape: {spectrogram.size()}, Label shape: {label.size()}")

                spectrogram = spectrogram.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                outputs = model(spectrogram)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                print(f"Batch {i+1} Loss: {loss.item()}")
            else:
                print(f"Skipping invalid data in batch {i+1} of epoch {epoch+1}")

        if num_batches > 0:
            average_loss = epoch_loss / num_batches
            print(f'Epoch {epoch+1}, Loss: {average_loss}')
        else:
            print(f"No valid batches in epoch {epoch+1}")

if __name__ == "__main__":
    dataset = EmotionDataset(root_dir=r"C:\Users\rashe\OneDrive\Desktop\speech recognition\myenv\audio_speech_actors_01-24")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = EmotionRecognitionModel(num_classes=len(observed_emotions))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_epochs = 20
    train_model(model, train_loader, device, num_epochs)

    # Save the trained model to a .pth file
    model_save_path = r"C:\Users\rashe\OneDrive\Desktop\speech recognition\my_trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
