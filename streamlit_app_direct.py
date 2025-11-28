import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from typing import Dict, Any
from io import BytesIO

# ==================== Google Drive Direct Download ====================
# Replace these with actual file IDs from your Google Drive
# You can get file ID by: right-click file > Get link > extract ID from URL

# Format: https://drive.google.com/file/d/[FILE_ID]/view
FILE_IDS = {
    'clf': 'YOUR_CLASSIFIER_FILE_ID',  # Replace with actual ID
    'regressor': 'YOUR_REGRESSOR_FILE_ID'  # Replace with actual ID
}

CACHE_DIR = os.path.expanduser('~/.mpce_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def download_from_gdrive(file_id: str, output_path: str) -> bool:
    """Download file directly from Google Drive using file ID."""
    try:
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        session = requests.Session()
        response = session.get(url, allow_redirects=True)
        
        # Handle large files with confirmation token
        if 'Content-Disposition' in response.headers:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            st.error(f"Failed to download: {file_id}")
            return False
    except Exception as e:
        st.error(f"Download error: {e}")
        return False


@st.cache_resource
def load_models():
    """Load models with proper error handling."""
    clf_path = os.path.join(CACHE_DIR, 'classifier.pkl')
    reg_path = os.path.join(CACHE_DIR, 'regressor.pkl')
    
    # Try to load from cache
    if os.path.exists(clf_path) and os.path.exists(reg_path):
        try:
            st.info("✅ Loading cached models...")
            clf_data = joblib.load(clf_path)
            reg_data = joblib.load(reg_path)
            return clf_data, reg_data
        except Exception as e:
            st.warning(f"Cache corrupted: {e}")
    
    # Download models
    st.info("📥 Downloading models (first time only)...")
    
    if FILE_IDS['clf'] == 'YOUR_CLASSIFIER_FILE_ID':
        st.error("⚠️ **Setup Required**: Please update FILE_IDS with actual Google Drive file IDs")
        st.write("""
        To get file IDs:
        1. Go to your Google Drive
        2. Right-click the model file → Get link
        3. Copy the ID from the link: `https://drive.google.com/file/d/[FILE_ID]/view`
        4. Update FILE_IDS in the code
        """)
        return None, None
    
    # Download classifier
    st.write("Downloading classifier...")
    if not download_from_gdrive(FILE_IDS['clf'], clf_path):
        return None, None
    
    # Download regressor
    st.write("Downloading regressor...")
    if not download_from_gdrive(FILE_IDS['regressor'], reg_path):
        return None, None
    
    try:
        clf_data = joblib.load(clf_path)
        reg_data = joblib.load(reg_path)
        st.success("✅ Models loaded successfully!")
        return clf_data, reg_data
    except Exception as e:
        st.error(f"Failed to load downloaded models: {e}")
        return None, None


# ==================== Setup ====================
st.set_page_config(page_title="MPCE Prediction", layout="wide")
st.title("🏠 MPCE Household Prediction Pipeline")

# Load models
clf_data, reg_data = load_models()

if clf_data is None or reg_data is None:
    st.stop()

# Extract models
try:
    regressors_raw = reg_data.get("models", {})
    regressors = {int(k): v for k, v in regressors_raw.items()}
    feature_info = reg_data.get("feature_info", {})
    
    cat_cols = feature_info.get("categorical_cols", [])
    num_cols = feature_info.get("numerical_cols", [])
    encoders = feature_info.get("encoders", {})
    scaler = feature_info.get("scaler")
except Exception as e:
    st.error(f"Model parsing failed: {e}")
    st.stop()


# ==================== Feature Processing ====================
def preprocess_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input features."""
    encoded = []
    
    for col in cat_cols:
        if col in raw_df and col in encoders:
            encoded.append(encoders[col].transform(raw_df[[col]]))
    
    if num_cols and scaler:
        scaled = scaler.transform(raw_df[num_cols])
        encoded.append(scaled)
    
    if encoded:
        return pd.concat([pd.DataFrame(e) for e in encoded], axis=1)
    return pd.DataFrame()


# ==================== UI ====================
st.write("Enter household details below:")

col1, col2 = st.columns(2)

with col1:
    sector = st.selectbox("Sector", [1, 2], format_func=lambda x: "Rural" if x == 1 else "Urban")
    state = st.number_input("State", min_value=1, value=1)
    nss_region = st.number_input("NSS Region", min_value=1, value=1)
    district = st.number_input("District", min_value=1, value=1)
    household_type = st.number_input("Household Type", min_value=1, value=1)

with col2:
    religion = st.number_input("Religion", min_value=1, value=1)
    social_group = st.number_input("Social Group", min_value=1, value=1)
    hh_size = st.number_input("HH Size", min_value=1, value=4)
    person_count = st.number_input("Person Count", min_value=1, value=4)
    avg_age = st.number_input("Average Age", min_value=0.0, value=35.0)

# Additional features...
max_age = st.number_input("Max Age", min_value=0, value=70)
min_age = st.number_input("Min Age", min_value=0, value=5)
gender_1_count = st.number_input("Gender 1 Count", min_value=0, value=2)
gender_2_count = st.number_input("Gender 2 Count", min_value=0, value=2)
gender_3_count = st.number_input("Gender 3 Count", min_value=0, value=0)
nco_3d = st.number_input("NCO_3D", min_value=0, value=0)
nic_5d = st.number_input("NIC_5D", min_value=0, value=0)
avg_education = st.number_input("Avg Education", min_value=0.0, value=8.0)
max_education = st.number_input("Max Education", min_value=0.0, value=12.0)

col3, col4 = st.columns(2)
with col3:
    meals_day_sum = st.number_input("Meals/day Sum", min_value=0.0, value=3.0)
    meals_day_mean = st.number_input("Meals/day Mean", min_value=0.0, value=3.0)
    meals_school_sum = st.number_input("School Meals Sum", min_value=0.0, value=0.0)
    meals_school_mean = st.number_input("School Meals Mean", min_value=0.0, value=0.0)
    meals_employer_sum = st.number_input("Employer Meals Sum", min_value=0.0, value=0.0)
with col4:
    meals_employer_mean = st.number_input("Employer Meals Mean", min_value=0.0, value=0.0)
    meals_payment_sum = st.number_input("Paid Meals Sum", min_value=0.0, value=0.0)
    meals_payment_mean = st.number_input("Paid Meals Mean", min_value=0.0, value=0.0)
    meals_home_sum = st.number_input("Home Meals Sum", min_value=0.0, value=3.0)
    meals_home_mean = st.number_input("Home Meals Mean", min_value=0.0, value=3.0)

internet_users_count = st.number_input("Internet Users", min_value=0, value=0)

# Online purchases
st.subheader("Online Purchases (Last 365 days)")
col5, col6, col7 = st.columns(3)

with col5:
    is_online_clothing = st.checkbox("Clothing", value=False)
    is_online_footwear = st.checkbox("Footwear", value=False)
    is_online_furniture = st.checkbox("Furniture", value=False)
    is_online_mobile = st.checkbox("Mobile", value=False)

with col6:
    is_online_personal = st.checkbox("Personal Goods", value=False)
    is_online_recreation = st.checkbox("Recreation", value=False)
    is_online_appliances = st.checkbox("Appliances", value=False)
    is_online_crockery = st.checkbox("Crockery", value=False)

with col7:
    is_online_sports = st.checkbox("Sports", value=False)
    is_online_medical = st.checkbox("Medical", value=False)
    is_online_bedding = st.checkbox("Bedding", value=False)

# Household assets
st.subheader("Household Assets")
col8, col9, col10 = st.columns(3)

with col8:
    is_tv = st.checkbox("TV", value=False)
    is_radio = st.checkbox("Radio", value=False)
    is_laptop = st.checkbox("Laptop", value=False)
    is_mobile_handset = st.checkbox("Mobile Handset", value=False)

with col9:
    is_bicycle = st.checkbox("Bicycle", value=False)
    is_motorcycle = st.checkbox("Motorcycle", value=False)
    is_motorcar = st.checkbox("Motor Car", value=False)
    is_trucks = st.checkbox("Trucks", value=False)

with col10:
    is_animal_cart = st.checkbox("Animal Cart", value=False)
    is_refrigerator = st.checkbox("Refrigerator", value=False)
    is_washing_machine = st.checkbox("Washing Machine", value=False)
    is_ac = st.checkbox("AC/Cooler", value=False)

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
            "Is_online_Clothing_Purchased_Last365": int(is_online_clothing),
            "Is_online_Footwear_Purchased_Last365": int(is_online_footwear),
            "Is_online_Furniture_fixturesPurchased_Last365": int(is_online_furniture),
            "Is_online_Mobile_Handset_Purchased_Last365": int(is_online_mobile),
            "Is_online_Personal_Goods_Purchased_Last365": int(is_online_personal),
            "Is_online_Recreation_Goods_Purchased_Last365": int(is_online_recreation),
            "Is_online_Household_Appliances_Purchased_Last365": int(is_online_appliances),
            "Is_online_Crockery_Utensils_Purchased_Last365": int(is_online_crockery),
            "Is_online_Sports_Goods_Purchased_Last365": int(is_online_sports),
            "Is_online_Medical_Equipment_Purchased_Last365": int(is_online_medical),
            "Is_online_Bedding_Purchased_Last365": int(is_online_bedding),
            "Is_HH_Have_Television": int(is_tv),
            "Is_HH_Have_Radio": int(is_radio),
            "Is_HH_Have_Laptop_PC": int(is_laptop),
            "Is_HH_Have_Mobile_handset": int(is_mobile_handset),
            "Is_HH_Have_Bicycle": int(is_bicycle),
            "Is_HH_Have_Motorcycle_scooter": int(is_motorcycle),
            "Is_HH_Have_Motorcar_jeep_van": int(is_motorcar),
            "Is_HH_Have_Trucks": int(is_trucks),
            "Is_HH_Have_Animal_cart": int(is_animal_cart),
            "Is_HH_Have_Refrigerator": int(is_refrigerator),
            "Is_HH_Have_Washing_machine": int(is_washing_machine),
            "Is_HH_Have_Airconditioner_aircooler": int(is_ac),
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
        
        if processed_df is None or processed_df.empty:
            st.error("Feature processing failed")
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
                    st.metric("Predicted MPCE", f"₹{predicted_mpce:,.2f}")
                with col_r2:
                    st.metric("Sector", sector_name)
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        import traceback
        st.write("Debug info:")
        st.code(traceback.format_exc())
