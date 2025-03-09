import zipfile
import os
zip_file_path = r"C:\Users\rashe\Downloads\archive.zip"
extraction_dir = r"C:\Users\rashe\OneDrive\Desktop\speech recognition\myenv"
os.makedirs(extraction_dir, exist_ok=True)
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)
print(f'Files extracted to {extraction_dir}')
