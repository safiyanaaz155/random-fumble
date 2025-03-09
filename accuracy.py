import torch

def test_model_accuracy(model, test_loader, device):
    """
    Function to test the accuracy of the trained model.

    Args:
    - model (torch.nn.Module): The trained emotion recognition model.
    - test_loader (DataLoader): DataLoader for the test or validation dataset.
    - device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
    - float: The accuracy of the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for faster performance
        for spectrogram, label in test_loader:
            if spectrogram is not None and label is not None:
                spectrogram = spectrogram.to(device)
                label = label.to(device)

                # Forward pass to get predictions
                outputs = model(spectrogram)

                # Get the predicted class (with the highest score)
                _, predicted = torch.max(outputs, 1)

                # Update counts
                correct_predictions += (predicted == label).sum().item()
                total_samples += label.size(0)

    # Calculate accuracy
    accuracy = (correct_predictions / total_samples) * 100
    return accuracy

if __name__ == "__main__":
    # Create a test dataset and DataLoader
    test_dataset = EmotionDataset(root_dir=r"C:\Users\rashe\OneDrive\Desktop\speech recognition\myenv\audio_speech_actors_01-24")  # Change this to your test dataset path
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load the trained model
    model = EmotionRecognitionModel(num_classes=len(observed_emotions))
    model.load_state_dict(torch.load(r"C:\Users\rashe\OneDrive\Desktop\speech recognition\my_trained_model.pth"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Calculate the model accuracy
    accuracy = test_model_accuracy(model, test_loader, device)
    print(f"Model Accuracy: {accuracy:.2f}%")
