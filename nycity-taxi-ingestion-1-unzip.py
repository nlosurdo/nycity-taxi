# Databricks notebook source
import zipfile
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the source folder from environment variable
source_folder = os.getenv('SOURCE_FOLDER')

# Check if the directory exists
if os.path.exists(source_folder):
    # Find zip files in the folder
    zip_files = [f for f in os.listdir(source_folder) if f.endswith('.zip')]

    if zip_files:
        # Unzip files if any are found
        for zip_file in zip_files:
            with zipfile.ZipFile(os.path.join(source_folder, zip_file), 'r') as zip_ref:
                zip_ref.extractall(source_folder)
            print(f"Unzipped {zip_file} successfully.")
    else:
        print("No zip files found. Nothing to unzip.")
else:
    print(f"The directory {source_folder} does not exist.")
