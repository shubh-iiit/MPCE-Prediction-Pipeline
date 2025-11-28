import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any
import urllib.request
import zipfile
import shutil

# ==================== Model Download & Cache ====================
CACHE_DIR = os.path.expanduser('~/.mpce_cache')
MODELS_DIR = os.path.join(CACHE_DIR, 'models')

# Direct download links (converted from Google Drive folder IDs)
# These are direct download URLs - update these with actual file URLs
MODEL_FILES = {
    'clf': {
        'name': 'sector_income_classifiers_tuned.pkl',
        'drive_id': '1ekW53Y1r4ga1h5YawMIMKmmjcTwKvf6A'
    },
    'regressor': {
        'name': 'sector_income_randomforestmodel.pkl',
        'drive_id': '1JgRdNFGw_K7-7yS9NO08Hdrr-_F4bDsa'
    }
}


def get_drive_download_url(drive_id: str, file_id: str) -> str:
    """Convert Google Drive file ID to direct download URL."""
    return f"https://drive.google.com/uc?id={file_id}&export=download"


@st.cache_resource
def load_models_with_fallback():
    """Load models with fallback to manual upload if auto-download fails."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    clf_path = os.path.join(MODELS_DIR, 'sector_income_classifiers_tuned.pkl')
    reg_path = os.path.join(MODELS_DIR, 'sector_income_randomforestmodel.pkl')
    
    # Check if models exist locally
    if os.path.exists(clf_path) and os.path.exists(reg_path):
        st.info("✅ Using cached models")
        try:
            clf_data = joblib.load(clf_path)
            reg_data = joblib.load(reg_path)
            return clf_data, reg_data
        except Exception as e:
            st.warning(f"Failed to load cached models: {e}")
    
    # Try to download from Google Drive
    st.info("📥 Attempting to download models from Google Drive...")
    try:
        import gdown
        
        # Download classifier
        clf_drive_id = '1yYz_xS7K8abcXYZ'  # You need actual file IDs
        st.write("Downloading classifier...")
        gdown.download(f'https://drive.google.com/uc?id={clf_drive_id}', clf_path, quiet=False)
        
        # Download regressor
        reg_drive_id = '1xBcD_9EfghIJKL'  # You need actual file IDs
        st.write("Downloading regressor...")
        gdown.download(f'https://drive.google.com/uc?id={reg_drive_id}', reg_path, quiet=False)
        
        clf_data = joblib.load(clf_path)
        reg_data = joblib.load(reg_path)
        return clf_data, reg_data
        
    except Exception as e:
        st.warning(f"⚠️ Auto-download failed: {e}")
        st.error("""
        **Models not found. Two options:**
        
        1. **Upload files manually** (you can add file upload widget below)
        2. **Contact developer** for direct download link
        
        Models needed:
        - `sector_income_classifiers_tuned.pkl`
        - `sector_income_randomforestmodel.pkl`
        """)
        return None, None


# ==================== Setup ====================
st.set_page_config(page_title="MPCE Prediction", layout="wide")
st.title("🏠 MPCE Household Prediction")

# Try to load models
clf_data, reg_data = load_models_with_fallback()

if clf_data is None or reg_data is None:
    st.stop()

# Extract models
try:
    regressors_raw = reg_data.get("models", {})
    regressors = {int(k): v for k, v in regressors_raw.items()}
    feature_info = reg_data.get("feature_info", {})
    classifiers = clf_data
    
    cat_cols = feature_info.get("categorical_cols", [])
    num_cols = feature_info.get("numerical_cols", [])
    encoders = feature_info.get("encoders", {})
    scaler = feature_info.get("scaler")
except Exception as e:
    st.error(f"Failed to parse models: {e}")
    st.stop()


# ==================== Feature Processing ====================
def preprocess_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input features using loaded encoders and scaler."""
    encoded = []
    
    # Encode categorical features
    for col in cat_cols:
        if col in raw_df and col in encoders:
            try:
                encoded.append(encoders[col].transform(raw_df[[col]]))
            except Exception as e:
                st.error(f"Error encoding {col}: {e}")
                return None
    
    # Scale numerical features
    if num_cols and scaler:
        try:
            scaled = scaler.transform(raw_df[num_cols])
            encoded.append(scaled)
        except Exception as e:
            st.error(f"Error scaling features: {e}")
            return None
    
    if encoded:
        return pd.concat([pd.DataFrame(e) for e in encoded], axis=1)
    return pd.DataFrame()


# ==================== UI ====================
st.write("Enter household details to predict Monthly Per Capita Expenditure (MPCE).")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Location & Demographics")
    sector = st.selectbox("Sector", [1, 2], format_func=lambda x: "Rural" if x == 1 else "Urban")
    state = st.number_input("State (code)", min_value=1, value=1)
    nss_region = st.number_input("NSS-Region", min_value=1, value=1)
    district = st.number_input("District", min_value=1, value=1)
    household_type = st.number_input("Household Type", min_value=1, value=1)
    religion = st.number_input("Religion (code)", min_value=1, value=1)
    social_group = st.number_input("Social Group (code)", min_value=1, value=1)
    hh_size = st.number_input("HH Size", min_value=1, value=4)

with col2:
    st.subheader("Person Details")
    person_count = st.number_input("Person Count", min_value=1, value=4)
    avg_age = st.number_input("Avg Age", min_value=0.0, value=35.0)
    max_age = st.number_input("Max Age", min_value=0, value=70)
    min_age = st.number_input("Min Age", min_value=0, value=5)
    gender_1_count = st.number_input("Gender 1", min_value=0, value=2)
    gender_2_count = st.number_input("Gender 2", min_value=0, value=2)
    gender_3_count = st.number_input("Gender 3", min_value=0, value=0)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Employment & Education")
    nco_3d = st.number_input("NCO_3D", min_value=0, value=0)
    nic_5d = st.number_input("NIC_5D", min_value=0, value=0)
    avg_education = st.number_input("Avg Education", min_value=0.0, value=8.0)
    max_education = st.number_input("Max Education", min_value=0.0, value=12.0)

with col4:
    st.subheader("Meals & Internet")
    meals_day_sum = st.number_input("Meals/day Sum", min_value=0.0, value=3.0)
    meals_day_mean = st.number_input("Meals/day Mean", min_value=0.0, value=3.0)
    meals_school_sum = st.number_input("School Meals Sum", min_value=0.0, value=0.0)
    meals_school_mean = st.number_input("School Meals Mean", min_value=0.0, value=0.0)
    meals_employer_sum = st.number_input("Employer Meals Sum", min_value=0.0, value=0.0)
    meals_employer_mean = st.number_input("Employer Meals Mean", min_value=0.0, value=0.0)
    meals_payment_sum = st.number_input("Paid Meals Sum", min_value=0.0, value=0.0)
    meals_payment_mean = st.number_input("Paid Meals Mean", min_value=0.0, value=0.0)
    meals_home_sum = st.number_input("Home Meals Sum", min_value=0.0, value=3.0)
    meals_home_mean = st.number_input("Home Meals Mean", min_value=0.0, value=3.0)
    internet_users_count = st.number_input("Internet Users", min_value=0, value=0)

st.subheader("Online Purchases")
col5, col6 = st.columns(2)

with col5:
    is_online_clothing = st.selectbox("Clothing", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_online_footwear = st.selectbox("Footwear", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_online_furniture = st.selectbox("Furniture", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_online_mobile = st.selectbox("Mobile", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_online_personal = st.selectbox("Personal", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col6:
    is_online_recreation = st.selectbox("Recreation", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_online_appliances = st.selectbox("Appliances", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_online_crockery = st.selectbox("Crockery", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_online_sports = st.selectbox("Sports", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_online_medical = st.selectbox("Medical", [0, 1], format_func=lambda x: "Yes" if x else "No")

st.subheader("Household Assets")
col7, col8, col9 = st.columns(3)

with col7:
    is_tv = st.selectbox("TV", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_radio = st.selectbox("Radio", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_laptop = st.selectbox("Laptop", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_mobile_handset = st.selectbox("Mobile", [0, 1], format_func=lambda x: "Yes" if x else "No", key="mobile_asset")
    is_bicycle = st.selectbox("Bicycle", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col8:
    is_motorcycle = st.selectbox("Motorcycle", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_motorcar = st.selectbox("Motor Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_trucks = st.selectbox("Trucks", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_animal_cart = st.selectbox("Animal Cart", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col9:
    is_refrigerator = st.selectbox("Refrigerator", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_washing_machine = st.selectbox("Washing Machine", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_ac = st.selectbox("AC/Cooler", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_online_bedding = st.selectbox("Online Bedding", [0, 1], format_func=lambda x: "Yes" if x else "No")

# ==================== Prediction ====================
if st.button("🔮 Predict MPCE", use_container_width=True):
    try:
        input_data = {
            "Sector": sector,
            "State": state,
            "NSS-Region": nss_region,
            "District": district,
            "Household Type": household_type,
            "Religion of the head of the household": religion,
            "Social Group of the head of the household": social_group,
            "HH Size (For FDQ)": hh_size,
            "NCO_3D": nco_3d,
            "NIC_5D": nic_5d,
            "Is_online_Clothing_Purchased_Last365": is_online_clothing,
            "Is_online_Footwear_Purchased_Last365": is_online_footwear,
            "Is_online_Furniture_fixturesPurchased_Last365": is_online_furniture,
            "Is_online_Mobile_Handset_Purchased_Last365": is_online_mobile,
            "Is_online_Personal_Goods_Purchased_Last365": is_online_personal,
            "Is_online_Recreation_Goods_Purchased_Last365": is_online_recreation,
            "Is_online_Household_Appliances_Purchased_Last365": is_online_appliances,
            "Is_online_Crockery_Utensils_Purchased_Last365": is_online_crockery,
            "Is_online_Sports_Goods_Purchased_Last365": is_online_sports,
            "Is_online_Medical_Equipment_Purchased_Last365": is_online_medical,
            "Is_online_Bedding_Purchased_Last365": is_online_bedding,
            "Is_HH_Have_Television": is_tv,
            "Is_HH_Have_Radio": is_radio,
            "Is_HH_Have_Laptop_PC": is_laptop,
            "Is_HH_Have_Mobile_handset": is_mobile_handset,
            "Is_HH_Have_Bicycle": is_bicycle,
            "Is_HH_Have_Motorcycle_scooter": is_motorcycle,
            "Is_HH_Have_Motorcar_jeep_van": is_motorcar,
            "Is_HH_Have_Trucks": is_trucks,
            "Is_HH_Have_Animal_cart": is_animal_cart,
            "Is_HH_Have_Refrigerator": is_refrigerator,
            "Is_HH_Have_Washing_machine": is_washing_machine,
            "Is_HH_Have_Airconditioner_aircooler": is_ac,
            "person_count": person_count,
            "avg_age": avg_age,
            "max_age": max_age,
            "min_age": min_age,
            "gender_1_count": gender_1_count,
            "gender_2_count": gender_2_count,
            "gender_3_count": gender_3_count,
            "avg_education": avg_education,
            "max_education": max_education,
            "No. of meals usually taken in a day_sum": meals_day_sum,
            "No. of meals usually taken in a day_mean": meals_day_mean,
            "No. of meals taken during last 30 days from school, balwadi etc._sum": meals_school_sum,
            "No. of meals taken during last 30 days from school, balwadi etc._mean": meals_school_mean,
            "No. of meals taken during last 30 days from employer as perquisites or part of wage_sum": meals_employer_sum,
            "No. of meals taken during last 30 days from employer as perquisites or part of wage_mean": meals_employer_mean,
            "No. of meals taken during last 30 days on payment_sum": meals_payment_sum,
            "No. of meals taken during last 30 days on payment_mean": meals_payment_mean,
            "No. of meals taken during last 30 days at home_sum": meals_home_sum,
            "No. of meals taken during last 30 days at home_mean": meals_home_mean,
            "internet_users_count": internet_users_count
        }
        
        input_df = pd.DataFrame([input_data])
        processed_df = preprocess_features(input_df)
        
        if processed_df is None:
            st.error("Failed to process features")
        else:
            regressor = regressors.get(sector)
            if regressor is None:
                st.error(f"No model for sector {sector}")
            else:
                predicted_mpce = regressor.predict(processed_df)[0]
                sector_name = "Rural" if sector == 1 else "Urban"
                
                st.success("✅ Prediction Complete!")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("Predicted MPCE", f"₹{predicted_mpce:.2f}")
                with col_r2:
                    st.metric("Sector", sector_name)
                    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
