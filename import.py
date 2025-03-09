import os

file_path = r"C:\Users\rashe\OneDrive\Desktop\speech recognition\my_trained_model.pth"
if os.path.exists(file_path):
    print("File exists!")
else:
    print("File does not exist.")
