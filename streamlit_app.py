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
    """Preprocess input features."""
    try:
        encoded = []
        for col in cat_cols:
            if col in raw_df and col in encoders:
                encoded.append(encoders[col].transform(raw_df[[col]]))
        if num_cols and scaler:
            encoded.append(scaler.transform(raw_df[num_cols]))
        if encoded:
            return pd.concat([pd.DataFrame(e) for e in encoded], axis=1)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None


# ==================== UI ====================
st.markdown("---")
st.write("Enter household details to predict MPCE")

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
    avg_age = st.number_input("Avg Age", min_value=0.0, value=35.0)

max_age = st.number_input("Max Age", min_value=0, value=70)
min_age = st.number_input("Min Age", min_value=0, value=5)
gender_1_count = st.number_input("Gender 1", min_value=0, value=2)
gender_2_count = st.number_input("Gender 2", min_value=0, value=2)
gender_3_count = st.number_input("Gender 3", min_value=0, value=0)
nco_3d = st.number_input("NCO_3D", min_value=0, value=0)
nic_5d = st.number_input("NIC_5D", min_value=0, value=0)
avg_education = st.number_input("Avg Education", min_value=0.0, value=8.0)
max_education = st.number_input("Max Education", min_value=0.0, value=12.0)

meals_day_sum = st.number_input("Meals/Day Sum", min_value=0.0, value=3.0)
meals_day_mean = st.number_input("Meals/Day Mean", min_value=0.0, value=3.0)
meals_school_sum = st.number_input("School Meals Sum", min_value=0.0, value=0.0)
meals_school_mean = st.number_input("School Meals Mean", min_value=0.0, value=0.0)
meals_employer_sum = st.number_input("Employer Meals Sum", min_value=0.0, value=0.0)
meals_employer_mean = st.number_input("Employer Meals Mean", min_value=0.0, value=0.0)
meals_payment_sum = st.number_input("Paid Meals Sum", min_value=0.0, value=0.0)
meals_payment_mean = st.number_input("Paid Meals Mean", min_value=0.0, value=0.0)
meals_home_sum = st.number_input("Home Meals Sum", min_value=0.0, value=3.0)
meals_home_mean = st.number_input("Home Meals Mean", min_value=0.0, value=3.0)
internet_users_count = st.number_input("Internet Users", min_value=0, value=0)

col3, col4, col5 = st.columns(3)

with col3:
    is_online_clothing = st.checkbox("Clothing", value=False)
    is_online_footwear = st.checkbox("Footwear", value=False)
    is_online_furniture = st.checkbox("Furniture", value=False)
    is_online_mobile = st.checkbox("Mobile Online", value=False)

with col4:
    is_online_personal = st.checkbox("Personal", value=False)
    is_online_recreation = st.checkbox("Recreation", value=False)
    is_online_appliances = st.checkbox("Appliances", value=False)
    is_online_crockery = st.checkbox("Crockery", value=False)

with col5:
    is_online_sports = st.checkbox("Sports", value=False)
    is_online_medical = st.checkbox("Medical", value=False)
    is_online_bedding = st.checkbox("Bedding", value=False)

col6, col7, col8 = st.columns(3)

with col6:
    is_tv = st.checkbox("TV", value=False)
    is_radio = st.checkbox("Radio", value=False)
    is_laptop = st.checkbox("Laptop", value=False)
    is_mobile_handset = st.checkbox("Mobile Asset", value=False)

with col7:
    is_bicycle = st.checkbox("Bicycle", value=False)
    is_motorcycle = st.checkbox("Motorcycle", value=False)
    is_motorcar = st.checkbox("Motor Car", value=False)
    is_trucks = st.checkbox("Trucks", value=False)

with col8:
    is_animal_cart = st.checkbox("Animal Cart", value=False)
    is_refrigerator = st.checkbox("Refrigerator", value=False)
    is_washing_machine = st.checkbox("Washing Machine", value=False)
    is_ac = st.checkbox("AC/Cooler", value=False)

# ==================== Prediction ====================
if st.button("🔮 Predict MPCE", use_container_width=True):
    try:
        input_data = {
            "Sector": sector, "State": state, "NSS-Region": nss_region, "District": district,
            "Household Type": household_type, "Religion of the head of the household": religion,
            "Social Group of the head of the household": social_group, "HH Size (For FDQ)": hh_size,
            "NCO_3D": nco_3d, "NIC_5D": nic_5d,
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
            "Is_HH_Have_Television": int(is_tv), "Is_HH_Have_Radio": int(is_radio),
            "Is_HH_Have_Laptop_PC": int(is_laptop), "Is_HH_Have_Mobile_handset": int(is_mobile_handset),
            "Is_HH_Have_Bicycle": int(is_bicycle), "Is_HH_Have_Motorcycle_scooter": int(is_motorcycle),
            "Is_HH_Have_Motorcar_jeep_van": int(is_motorcar), "Is_HH_Have_Trucks": int(is_trucks),
            "Is_HH_Have_Animal_cart": int(is_animal_cart), "Is_HH_Have_Refrigerator": int(is_refrigerator),
            "Is_HH_Have_Washing_machine": int(is_washing_machine),
            "Is_HH_Have_Airconditioner_aircooler": int(is_ac),
            "person_count": person_count, "avg_age": avg_age, "max_age": max_age, "min_age": min_age,
            "gender_1_count": gender_1_count, "gender_2_count": gender_2_count, "gender_3_count": gender_3_count,
            "avg_education": avg_education, "max_education": max_education,
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
                    st.metric("Predicted MPCE (₹)", f"{predicted_mpce:,.2f}")
                with col_r2:
                    st.metric("Sector", sector_name)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())


# ==================== Setup ====================
st.set_page_config(page_title="MPCE Prediction", layout="wide")
st.title("🏠 MPCE Household Prediction")

st.write("Loading models...")
clf_data, reg_data = load_models_from_huggingface()

if clf_data is None or reg_data is None:
    st.error("Could not load models. Check the logs above.")
    st.stop()

st.write("Parsing models...")
try:
    regressors_raw = reg_data.get("models", {})
    regressors = {int(k): v for k, v in regressors_raw.items()}
    feature_info = reg_data.get("feature_info", {})
    
    cat_cols = feature_info.get("categorical_cols", [])
    num_cols = feature_info.get("numerical_cols", [])
    encoders = feature_info.get("encoders", {})
    scaler = feature_info.get("scaler")
    
    st.write(f"✓ Found {len(regressors)} regressor models")
    st.write(f"✓ Found {len(cat_cols)} categorical columns")
    st.write(f"✓ Found {len(num_cols)} numerical columns")
except Exception as e:
    st.error(f"Failed to parse models: {e}")
    import traceback
    st.write(traceback.format_exc())
    st.stop()


def preprocess_features(raw_df):
    """Preprocess input features."""
    try:
        encoded = []
        for col in cat_cols:
            if col in raw_df and col in encoders:
                encoded.append(encoders[col].transform(raw_df[[col]]))
        if num_cols and scaler:
            encoded.append(scaler.transform(raw_df[num_cols]))
        if encoded:
            return pd.concat([pd.DataFrame(e) for e in encoded], axis=1)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None


# ==================== UI ====================
st.markdown("---")
st.write("Enter household details to predict MPCE")

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
    avg_age = st.number_input("Avg Age", min_value=0.0, value=35.0)

max_age = st.number_input("Max Age", min_value=0, value=70)
min_age = st.number_input("Min Age", min_value=0, value=5)
gender_1_count = st.number_input("Gender 1", min_value=0, value=2)
gender_2_count = st.number_input("Gender 2", min_value=0, value=2)
gender_3_count = st.number_input("Gender 3", min_value=0, value=0)
nco_3d = st.number_input("NCO_3D", min_value=0, value=0)
nic_5d = st.number_input("NIC_5D", min_value=0, value=0)
avg_education = st.number_input("Avg Education", min_value=0.0, value=8.0)
max_education = st.number_input("Max Education", min_value=0.0, value=12.0)

meals_day_sum = st.number_input("Meals/Day Sum", min_value=0.0, value=3.0)
meals_day_mean = st.number_input("Meals/Day Mean", min_value=0.0, value=3.0)
meals_school_sum = st.number_input("School Meals Sum", min_value=0.0, value=0.0)
meals_school_mean = st.number_input("School Meals Mean", min_value=0.0, value=0.0)
meals_employer_sum = st.number_input("Employer Meals Sum", min_value=0.0, value=0.0)
meals_employer_mean = st.number_input("Employer Meals Mean", min_value=0.0, value=0.0)
meals_payment_sum = st.number_input("Paid Meals Sum", min_value=0.0, value=0.0)
meals_payment_mean = st.number_input("Paid Meals Mean", min_value=0.0, value=0.0)
meals_home_sum = st.number_input("Home Meals Sum", min_value=0.0, value=3.0)
meals_home_mean = st.number_input("Home Meals Mean", min_value=0.0, value=3.0)
internet_users_count = st.number_input("Internet Users", min_value=0, value=0)

col3, col4, col5 = st.columns(3)

with col3:
    is_online_clothing = st.checkbox("Clothing", value=False)
    is_online_footwear = st.checkbox("Footwear", value=False)
    is_online_furniture = st.checkbox("Furniture", value=False)
    is_online_mobile = st.checkbox("Mobile Online", value=False)

with col4:
    is_online_personal = st.checkbox("Personal", value=False)
    is_online_recreation = st.checkbox("Recreation", value=False)
    is_online_appliances = st.checkbox("Appliances", value=False)
    is_online_crockery = st.checkbox("Crockery", value=False)

with col5:
    is_online_sports = st.checkbox("Sports", value=False)
    is_online_medical = st.checkbox("Medical", value=False)
    is_online_bedding = st.checkbox("Bedding", value=False)

col6, col7, col8 = st.columns(3)

with col6:
    is_tv = st.checkbox("TV", value=False)
    is_radio = st.checkbox("Radio", value=False)
    is_laptop = st.checkbox("Laptop", value=False)
    is_mobile_handset = st.checkbox("Mobile Asset", value=False)

with col7:
    is_bicycle = st.checkbox("Bicycle", value=False)
    is_motorcycle = st.checkbox("Motorcycle", value=False)
    is_motorcar = st.checkbox("Motor Car", value=False)
    is_trucks = st.checkbox("Trucks", value=False)

with col8:
    is_animal_cart = st.checkbox("Animal Cart", value=False)
    is_refrigerator = st.checkbox("Refrigerator", value=False)
    is_washing_machine = st.checkbox("Washing Machine", value=False)
    is_ac = st.checkbox("AC/Cooler", value=False)

# ==================== Prediction ====================
if st.button("🔮 Predict MPCE", use_container_width=True):
    try:
        input_data = {
            "Sector": sector, "State": state, "NSS-Region": nss_region, "District": district,
            "Household Type": household_type, "Religion of the head of the household": religion,
            "Social Group of the head of the household": social_group, "HH Size (For FDQ)": hh_size,
            "NCO_3D": nco_3d, "NIC_5D": nic_5d,
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
            "Is_HH_Have_Television": int(is_tv), "Is_HH_Have_Radio": int(is_radio),
            "Is_HH_Have_Laptop_PC": int(is_laptop), "Is_HH_Have_Mobile_handset": int(is_mobile_handset),
            "Is_HH_Have_Bicycle": int(is_bicycle), "Is_HH_Have_Motorcycle_scooter": int(is_motorcycle),
            "Is_HH_Have_Motorcar_jeep_van": int(is_motorcar), "Is_HH_Have_Trucks": int(is_trucks),
            "Is_HH_Have_Animal_cart": int(is_animal_cart), "Is_HH_Have_Refrigerator": int(is_refrigerator),
            "Is_HH_Have_Washing_machine": int(is_washing_machine),
            "Is_HH_Have_Airconditioner_aircooler": int(is_ac),
            "person_count": person_count, "avg_age": avg_age, "max_age": max_age, "min_age": min_age,
            "gender_1_count": gender_1_count, "gender_2_count": gender_2_count, "gender_3_count": gender_3_count,
            "avg_education": avg_education, "max_education": max_education,
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
                    st.metric("Predicted MPCE (₹)", f"{predicted_mpce:,.2f}")
                with col_r2:
                    st.metric("Sector", sector_name)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())


# ==================== UI ====================
st.write("Enter household details to predict Monthly Per Capita Expenditure (MPCE)")

col1, col2 = st.columns(2)

with col1:
    sector = st.selectbox("Sector", [1, 2], format_func=lambda x: "Rural" if x == 1 else "Urban")
    state = st.number_input("State", min_value=1, value=1)
    nss_region = st.number_input("NSS Region", min_value=1, value=1)
    district = st.number_input("District", min_value=1, value=1)
    household_type = st.number_input("Household Type", min_value=1, value=1)
    religion = st.number_input("Religion", min_value=1, value=1)
    social_group = st.number_input("Social Group", min_value=1, value=1)
    hh_size = st.number_input("HH Size", min_value=1, value=4)

with col2:
    person_count = st.number_input("Person Count", min_value=1, value=4)
    avg_age = st.number_input("Avg Age", min_value=0.0, value=35.0)
    max_age = st.number_input("Max Age", min_value=0, value=70)
    min_age = st.number_input("Min Age", min_value=0, value=5)
    gender_1_count = st.number_input("Gender 1", min_value=0, value=2)
    gender_2_count = st.number_input("Gender 2", min_value=0, value=2)
    gender_3_count = st.number_input("Gender 3", min_value=0, value=0)
    nco_3d = st.number_input("NCO_3D", min_value=0, value=0)

nic_5d = st.number_input("NIC_5D", min_value=0, value=0)
avg_education = st.number_input("Avg Education", min_value=0.0, value=8.0)
max_education = st.number_input("Max Education", min_value=0.0, value=12.0)

meals_day_sum = st.number_input("Meals/Day Sum", min_value=0.0, value=3.0)
meals_day_mean = st.number_input("Meals/Day Mean", min_value=0.0, value=3.0)
meals_school_sum = st.number_input("School Meals Sum", min_value=0.0, value=0.0)
meals_school_mean = st.number_input("School Meals Mean", min_value=0.0, value=0.0)
meals_employer_sum = st.number_input("Employer Meals Sum", min_value=0.0, value=0.0)
meals_employer_mean = st.number_input("Employer Meals Mean", min_value=0.0, value=0.0)
meals_payment_sum = st.number_input("Paid Meals Sum", min_value=0.0, value=0.0)
meals_payment_mean = st.number_input("Paid Meals Mean", min_value=0.0, value=0.0)
meals_home_sum = st.number_input("Home Meals Sum", min_value=0.0, value=3.0)
meals_home_mean = st.number_input("Home Meals Mean", min_value=0.0, value=3.0)
internet_users_count = st.number_input("Internet Users", min_value=0, value=0)

col3, col4, col5 = st.columns(3)

with col3:
    is_online_clothing = st.checkbox("Clothing", value=False)
    is_online_footwear = st.checkbox("Footwear", value=False)
    is_online_furniture = st.checkbox("Furniture", value=False)
    is_online_mobile = st.checkbox("Mobile Online", value=False)

with col4:
    is_online_personal = st.checkbox("Personal", value=False)
    is_online_recreation = st.checkbox("Recreation", value=False)
    is_online_appliances = st.checkbox("Appliances", value=False)
    is_online_crockery = st.checkbox("Crockery", value=False)

with col5:
    is_online_sports = st.checkbox("Sports", value=False)
    is_online_medical = st.checkbox("Medical", value=False)
    is_online_bedding = st.checkbox("Bedding", value=False)

col6, col7, col8 = st.columns(3)

with col6:
    is_tv = st.checkbox("TV", value=False)
    is_radio = st.checkbox("Radio", value=False)
    is_laptop = st.checkbox("Laptop", value=False)
    is_mobile_handset = st.checkbox("Mobile Asset", value=False)

with col7:
    is_bicycle = st.checkbox("Bicycle", value=False)
    is_motorcycle = st.checkbox("Motorcycle", value=False)
    is_motorcar = st.checkbox("Motor Car", value=False)
    is_trucks = st.checkbox("Trucks", value=False)

with col8:
    is_animal_cart = st.checkbox("Animal Cart", value=False)
    is_refrigerator = st.checkbox("Refrigerator", value=False)
    is_washing_machine = st.checkbox("Washing Machine", value=False)
    is_ac = st.checkbox("AC/Cooler", value=False)

# ==================== Prediction ====================
if st.button("🔮 Predict MPCE", use_container_width=True):
    try:
        input_data = {
            "Sector": sector, "State": state, "NSS-Region": nss_region, "District": district,
            "Household Type": household_type, "Religion of the head of the household": religion,
            "Social Group of the head of the household": social_group, "HH Size (For FDQ)": hh_size,
            "NCO_3D": nco_3d, "NIC_5D": nic_5d,
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
            "Is_HH_Have_Television": int(is_tv), "Is_HH_Have_Radio": int(is_radio),
            "Is_HH_Have_Laptop_PC": int(is_laptop), "Is_HH_Have_Mobile_handset": int(is_mobile_handset),
            "Is_HH_Have_Bicycle": int(is_bicycle), "Is_HH_Have_Motorcycle_scooter": int(is_motorcycle),
            "Is_HH_Have_Motorcar_jeep_van": int(is_motorcar), "Is_HH_Have_Trucks": int(is_trucks),
            "Is_HH_Have_Animal_cart": int(is_animal_cart), "Is_HH_Have_Refrigerator": int(is_refrigerator),
            "Is_HH_Have_Washing_machine": int(is_washing_machine),
            "Is_HH_Have_Airconditioner_aircooler": int(is_ac),
            "person_count": person_count, "avg_age": avg_age, "max_age": max_age, "min_age": min_age,
            "gender_1_count": gender_1_count, "gender_2_count": gender_2_count, "gender_3_count": gender_3_count,
            "avg_education": avg_education, "max_education": max_education,
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
                    st.metric("Predicted MPCE (₹)", f"{predicted_mpce:,.2f}")
                with col_r2:
                    st.metric("Sector", sector_name)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
