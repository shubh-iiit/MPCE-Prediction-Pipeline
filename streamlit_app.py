import streamlit as st

st.write("✓ Step 1: Streamlit imported")

try:
    import pandas as pd
    st.write("✓ Step 2: pandas imported")
except Exception as e:
    st.error(f"✗ pandas failed: {e}")
    st.stop()

try:
    import joblib
    st.write("✓ Step 3: joblib imported")
except Exception as e:
    st.error(f"✗ joblib failed: {e}")
    st.stop()

try:
    import os
    st.write("✓ Step 4: os imported")
except Exception as e:
    st.error(f"✗ os failed: {e}")
    st.stop()

try:
    import warnings
    import pickle
    warnings.filterwarnings('ignore')
    st.write("✓ Step 5: warnings & pickle imported")
except Exception as e:
    st.error(f"✗ warnings/pickle failed: {e}")
    st.stop()

try:
    from huggingface_hub import hf_hub_download
    st.write("✓ Step 6: huggingface_hub imported")
except Exception as e:
    st.error(f"✗ huggingface_hub failed: {e}")
    st.stop()

st.write("---")
st.write("All imports successful! Configuring page...")

try:
    st.set_page_config(page_title="MPCE Prediction", layout="wide")
    st.title("🏠 MPCE Household Prediction")
    st.write("Page config done!")
except Exception as e:
    st.error(f"✗ Page config failed: {e}")
    st.stop()

st.write("Setting up cache directory...")

try:
    MODEL_REPO = "shubh-iiit/mpce-models"
    CACHE_DIR = os.path.expanduser('~/.mpce_cache')
    os.makedirs(CACHE_DIR, exist_ok=True)
    st.write(f"✓ Cache dir ready: {CACHE_DIR}")
except Exception as e:
    st.error(f"✗ Cache dir failed: {e}")
    st.stop()

st.write("---")
st.write("Now attempting to download models...")

try:
    # Try to download classifier
    st.write("📥 Downloading classifier...")
    clf_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="models_clf/sector_income_classifiers_tuned.pkl",
        cache_dir=CACHE_DIR
    )
    st.write(f"✓ Classifier downloaded: {os.path.getsize(clf_path)} bytes")
    
    # Try to load classifier
    st.write("📦 Loading classifier with joblib...")
    clf_data = joblib.load(clf_path)
    st.write(f"✓ Classifier loaded: type={type(clf_data)}")
    if isinstance(clf_data, dict):
        st.write(f"  Dict keys: {list(clf_data.keys())}")
    
except Exception as e:
    st.error(f"✗ Classifier error: {e}")
    import traceback
    st.write(traceback.format_exc())
    st.stop()

try:
    # Try to download regressor
    st.write("📥 Downloading regressor...")
    reg_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="models_regressor/sector_income_randomforestmodel.pkl",
        cache_dir=CACHE_DIR
    )
    st.write(f"✓ Regressor downloaded: {os.path.getsize(reg_path)} bytes")
    
    # Try to load regressor
    st.write("📦 Loading regressor with joblib...")
    reg_data = joblib.load(reg_path)
    st.write(f"✓ Regressor loaded: type={type(reg_data)}")
    if isinstance(reg_data, dict):
        st.write(f"  Dict keys: {list(reg_data.keys())}")
    
except Exception as e:
    st.error(f"✗ Regressor error: {e}")
    import traceback
    st.write(traceback.format_exc())
    st.stop()

st.success("✅ All diagnostics complete!")
st.write("Share the output above with me to proceed!")
