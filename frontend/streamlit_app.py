import streamlit as st
import joblib
import os
import gdown
import pandas as pd
from typing import Dict, Any

# ==================== Google Drive Model Loading ====================
MODELS_BASE_DIR = os.path.expanduser('~/.mpce_models')
DRIVE_FOLDERS = {
    'clf': '1ekW53Y1r4ga1h5YawMIMKmmjcTwKvf6A',
    'regressor': '1JgRdNFGw_K7-7yS9NO08Hdrr-_F4bDsa'
}


@st.cache_resource
def download_and_load_models():
    """Download models from Google Drive and cache them."""
    clf_dir = os.path.join(MODELS_BASE_DIR, 'models_clf')
    regressor_dir = os.path.join(MODELS_BASE_DIR, 'models_regressor')
    
    # Download classifier models if needed
    if not os.path.exists(clf_dir) or not os.listdir(clf_dir):
        st.info("📥 Downloading classifier models from Google Drive...")
        os.makedirs(clf_dir, exist_ok=True)
        url = f'https://drive.google.com/drive/folders/{DRIVE_FOLDERS["clf"]}'
        gdown.download_folder(url, output=clf_dir, quiet=True)
    
    # Download regressor models if needed
    if not os.path.exists(regressor_dir) or not os.listdir(regressor_dir):
        st.info("📥 Downloading regressor models from Google Drive...")
        os.makedirs(regressor_dir, exist_ok=True)
        url = f'https://drive.google.com/drive/folders/{DRIVE_FOLDERS["regressor"]}'
        gdown.download_folder(url, output=regressor_dir, quiet=True)
    
    # Load classifier
    clf_path = os.path.join(clf_dir, 'sector_income_classifiers_tuned.pkl')
    clf_data = joblib.load(clf_path)
    
    # Load regressor
    reg_path = os.path.join(regressor_dir, 'sector_income_randomforestmodel.pkl')
    reg_data = joblib.load(reg_path)
    
    return clf_data, reg_data


# ==================== Load Models ====================
st.set_page_config(page_title="MPCE Prediction", layout="wide")

with st.spinner("Loading models..."):
    clf_data, reg_data = download_and_load_models()

# Extract data
regressors_raw: Dict[Any, Any] = reg_data["models"]
regressors: Dict[int, Any] = {int(k): v for k, v in regressors_raw.items()}
feature_info: Dict[str, Any] = reg_data["feature_info"]
classifiers = clf_data

cat_cols = feature_info["categorical_cols"]
num_cols = feature_info["numerical_cols"]
encoders = feature_info["encoders"]
scaler = feature_info["scaler"]


# ==================== Feature Preprocessing ====================
def get_expected_feature_order():
    feature_list = []
    feature_list.extend(num_cols)
    for col in cat_cols:
        cats = encoders[col].categories_[0]
        feature_list.extend([f"{col}_{cat}" for cat in cats])
    return feature_list


def preprocess_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = feature_info['categorical_cols']
    numerical_cols   = feature_info['numerical_cols']
    encoders         = feature_info['encoders']
    scaler           = feature_info['scaler']

    encoded = []
    # encode all categoricals
    for col in categorical_cols:
        if col in raw_df:
            encoded.append(encoders[col].transform(raw_df[[col]]))
    # scale numericals
    if numerical_cols:
        scaled = scaler.transform(raw_df[numerical_cols])
        encoded.append(scaled)

    if encoded:
        return pd.concat([pd.DataFrame(e) for e in encoded], axis=1)
    return pd.DataFrame()


# ==================== UI ====================
st.title("🏠 MPCE Household Prediction")
st.write("Enter household details to predict Monthly Per Capita Expenditure (MPCE).")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location & Demographics")
    sector = st.selectbox("Sector", [1, 2], format_func=lambda x: "Rural" if x == 1 else "Urban")
    state = st.number_input("State (code)", min_value=1, value=1)
    nss_region = st.number_input("NSS-Region", min_value=1, value=1)
    district = st.number_input("District", min_value=1, value=1)
    household_type = st.number_input("Household Type (code)", min_value=1, value=1)
    religion = st.number_input("Religion of head (code)", min_value=1, value=1)
    social_group = st.number_input("Social Group (code)", min_value=1, value=1)

with col2:
    st.subheader("👥 Household Composition")
    hh_size = st.number_input("HH Size", min_value=1, value=4)
    person_count = st.number_input("Person Count", min_value=1, value=4)
    avg_age = st.number_input("Average Age", min_value=0.0, value=30.0)
    max_age = st.number_input("Max Age", min_value=0, value=65)
    min_age = st.number_input("Min Age", min_value=0, value=5)
    gender_1_count = st.number_input("Gender 1 Count", min_value=0, value=2)
    gender_2_count = st.number_input("Gender 2 Count", min_value=0, value=2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("💼 Employment")
    nco_3d = st.number_input("NCO_3D", min_value=0, value=0)
    nic_5d = st.number_input("NIC_5D", min_value=0, value=0)
    avg_education = st.number_input("Average Education", min_value=0.0, value=8.0)
    max_education = st.number_input("Max Education", min_value=0.0, value=12.0)

with col4:
    st.subheader("🍽️ Meals & Internet")
    meals_day_sum = st.number_input("Meals/day (sum)", min_value=0.0, value=3.0)
    meals_day_mean = st.number_input("Meals/day (mean)", min_value=0.0, value=3.0)
    meals_school_sum = st.number_input("School meals (sum)", min_value=0.0, value=0.0)
    meals_school_mean = st.number_input("School meals (mean)", min_value=0.0, value=0.0)
    meals_employer_sum = st.number_input("Employer meals (sum)", min_value=0.0, value=0.0)
    meals_employer_mean = st.number_input("Employer meals (mean)", min_value=0.0, value=0.0)
    meals_payment_sum = st.number_input("Paid meals (sum)", min_value=0.0, value=0.0)
    meals_payment_mean = st.number_input("Paid meals (mean)", min_value=0.0, value=0.0)
    meals_home_sum = st.number_input("Home meals (sum)", min_value=0.0, value=3.0)
    meals_home_mean = st.number_input("Home meals (mean)", min_value=0.0, value=3.0)
    internet_users_count = st.number_input("Internet Users", min_value=0, value=0)

st.subheader("🛍️ Online Purchases (Last 365 days)")
col5, col6 = st.columns(2)

def binary_dropdown(label):
    return st.selectbox(label, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=label)

with col5:
    is_online_clothing = binary_dropdown("Clothing")
    is_online_footwear = binary_dropdown("Footwear")
    is_online_furniture = binary_dropdown("Furniture")
    is_online_mobile = binary_dropdown("Mobile Handset")
    is_online_personal = binary_dropdown("Personal Goods")
    is_online_recreation = binary_dropdown("Recreation Goods")

with col6:
    is_online_appliances = binary_dropdown("Household Appliances")
    is_online_crockery = binary_dropdown("Crockery/Utensils")
    is_online_sports = binary_dropdown("Sports Goods")
    is_online_medical = binary_dropdown("Medical Equipment")
    is_online_bedding = binary_dropdown("Bedding")

st.subheader("🏢 Household Possessions")
col7, col8, col9 = st.columns(3)

with col7:
    is_tv = binary_dropdown("Television")
    is_radio = binary_dropdown("Radio")
    is_laptop = binary_dropdown("Laptop/PC")
    is_mobile_handset = binary_dropdown("Mobile Handset (Possession)")
    is_bicycle = binary_dropdown("Bicycle")

with col8:
    is_motorcycle = binary_dropdown("Motorcycle/Scooter")
    is_motorcar = binary_dropdown("Motorcar/Jeep/Van")
    is_trucks = binary_dropdown("Trucks")
    is_animal_cart = binary_dropdown("Animal Cart")

with col9:
    is_refrigerator = binary_dropdown("Refrigerator")
    is_washing_machine = binary_dropdown("Washing Machine")
    is_ac = binary_dropdown("Air Conditioner/Cooler")

# ==================== Prediction ====================
if st.button("🔮 Predict MPCE", use_container_width=True):
    try:
        # Create input dataframe
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
            "gender_3_count": 0,
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
        
        # Get regressor for the sector
        regressor = regressors.get(sector)
        if regressor is None:
            st.error(f"❌ No regressor found for sector {sector}")
        else:
            predicted_mpce = regressor.predict(processed_df)[0]
            sector_name = "Rural" if sector == 1 else "Urban"
            
            st.success(f"✅ Prediction Complete!")
            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.metric("Predicted MPCE", f"₹{predicted_mpce:.2f}")
            with col_result2:
                st.metric("Sector", sector_name)
                
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.write(f"Debug info: {e}")

st.markdown("---")
st.markdown("**Note:** This model requires all input features. Ensure all fields are filled.")
