# Why gdown Fails on Streamlit Cloud & Solutions

## Problem
`gdown` folder downloads fail on Streamlit Cloud because:
1. **Headless environment**: No browser, can't handle Google Drive's authentication flow
2. **Cookie issues**: Doesn't store cookies properly for large files
3. **Timeout**: Folder downloads take too long
4. **Rate limiting**: Google Drive blocks automation

## Solutions

### Solution 1: Use Direct File IDs (Recommended)
Instead of folder download, use individual file IDs.

```python
# Get file ID:
# 1. Right-click file in Google Drive
# 2. Get link → copy ID from: https://drive.google.com/file/d/[FILE_ID]/view
# 3. Use direct download URL

import requests
def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(url, allow_redirects=True)
    with open(output_path, 'wb') as f:
        f.write(response.content)
```

**Advantages**: Works reliably, fast, no authentication needed

### Solution 2: Host Models Elsewhere
- **Hugging Face**: Free model hosting
- **AWS S3**: Direct file downloads
- **GitHub Releases**: Upload files as release artifacts
- **Firebase Storage**: Free tier available

### Solution 3: Package Models with Code
Add model files directly to Git LFS or separate branch:
```bash
git lfs track "*.pkl"
git add models/
git commit -m "Add model files"
```

## Implementation for Your Project

### Option A: Use `streamlit_app_direct.py`
1. Get actual file IDs from Google Drive
2. Update `FILE_IDS` in the code
3. Deploy

### Option B: Use Hugging Face
```python
import huggingface_hub

model = huggingface_hub.hf_hub_download(
    repo_id="your-username/mpce-models",
    filename="classifier.pkl"
)
```

### Option C: Use GitHub Releases
1. Upload `.pkl` files to GitHub Releases
2. Download with direct URL:
```python
import urllib.request
url = "https://github.com/user/repo/releases/download/v1.0/classifier.pkl"
urllib.request.urlretrieve(url, "classifier.pkl")
```

## Your Next Steps

1. **Find actual file IDs**:
   - Open https://drive.google.com/drive/folders/1ekW53Y1r4ga1h5YawMIMKmmjcTwKvf6A
   - Right-click each `.pkl` file → Get link
   - Extract ID from URL

2. **Update `streamlit_app_direct.py`**:
   ```python
   FILE_IDS = {
       'clf': 'YOUR_ACTUAL_CLASSIFIER_ID',
       'regressor': 'YOUR_ACTUAL_REGRESSOR_ID'
   }
   ```

3. **Push and deploy** on Streamlit Cloud with `streamlit_app_direct.py`

## Testing Locally
```bash
streamlit run streamlit_app_direct.py
```

The app will download models on first run and cache them locally.
