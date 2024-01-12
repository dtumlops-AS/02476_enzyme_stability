# Use the Kaggle module to download novozymes dataset
import kaggle
import zipfile
import os

# Parameters
raw_path = "mlops_enzyme_stability/data/raw/"

# Authenticate with Kaggle
try:
    kaggle.api.authenticate()
except Exception:
    print("Please provide a valid Kaggle username and API key")

# Download the dataset
try:
    kaggle.api.competition_download_files("novozymes-enzyme-stability-prediction", path=raw_path)
except Exception:
    print("Could not download dataset, please check that you have accepted the competition rules")
    
# Unzip the dataset
zipfile.ZipFile(raw_path + "novozymes-enzyme-stability-prediction.zip").extractall(raw_path)

# Remove the zip file
os.remove(raw_path + "novozymes-enzyme-stability-prediction.zip")
