import os
import joblib
import gdown
from typing import Dict, Any

# Google Drive folder IDs
DRIVE_FOLDERS = {
    'clf': '1ekW53Y1r4ga1h5YawMIMKmmjcTwKvf6A',
    'regressor': '1JgRdNFGw_K7-7yS9NO08Hdrr-_F4bDsa'
}

MODELS_BASE_DIR = os.path.expanduser('~/.mpce_models')


def download_folder_from_drive(folder_id: str, output_dir: str) -> None:
    """Download all files from a Google Drive folder."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    url = f'https://drive.google.com/drive/folders/{folder_id}'
    gdown.download_folder(url, output=output_dir, quiet=False)


def ensure_models_exist() -> None:
    """Ensure models are downloaded. Download if missing."""
    clf_dir = os.path.join(MODELS_BASE_DIR, 'models_clf')
    regressor_dir = os.path.join(MODELS_BASE_DIR, 'models_regressor')
    
    # Check if classifier models exist
    if not os.path.exists(clf_dir) or not os.listdir(clf_dir):
        print(f"Downloading classifier models to {clf_dir}...")
        download_folder_from_drive(DRIVE_FOLDERS['clf'], clf_dir)
    
    # Check if regressor models exist
    if not os.path.exists(regressor_dir) or not os.listdir(regressor_dir):
        print(f"Downloading regressor models to {regressor_dir}...")
        download_folder_from_drive(DRIVE_FOLDERS['regressor'], regressor_dir)


def load_models() -> tuple:
    """Load both classifier and regressor models."""
    ensure_models_exist()
    
    clf_path = os.path.join(MODELS_BASE_DIR, 'models_clf', 'sector_income_classifiers_tuned.pkl')
    reg_path = os.path.join(MODELS_BASE_DIR, 'models_regressor', 'sector_income_randomforestmodel.pkl')
    
    print(f"Loading classifier from {clf_path}...")
    clf_data = joblib.load(clf_path)
    
    print(f"Loading regressor from {reg_path}...")
    reg_data = joblib.load(reg_path)
    
    return clf_data, reg_data
