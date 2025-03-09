import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

project_dir = os.path.dirname(os.path.abspath(__file__)) 
dataset_dir = os.path.join(project_dir, "data")

os.makedirs(dataset_dir, exist_ok=True)
api.dataset_download_files("puneet6060/intel-image-classification", path=dataset_dir, unzip=True)

print("Dataset downloaded and extracted to:", dataset_dir)
print("Contents:", os.listdir(dataset_dir))
