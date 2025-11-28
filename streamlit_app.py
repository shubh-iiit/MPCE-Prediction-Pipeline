import streamlit as st
import pandas as pd
import joblib
import os
import warnings
import pickle
import gc

warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG (MUST BE FIRST) ====================
st.set_page_config(page_title="MPCE Prediction", layout="wide", initial_sidebar_state="collapsed")

# ==================== CONSTANTS ====================
MODEL_REPO = "shubh-iiit/mpce-models"
CACHE_DIR = os.path.expanduser('~/.mpce_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== LAZY MODEL LOADING WITH MEMORY MANAGEMENT ====================
class ModelLoader:
    """Lazy loader for models with memory cleanup."""
    _instance = None
    _models = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def get_models(self):
        """Get models, loading them if needed."""
        if self._models is not None:
            return self._models
        
        from huggingface_hub import hf_hub_download
        
        try:
            with st.spinner("📥 Loading models..."):
                # Download files
                clf_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename="models_clf/sector_income_classifiers_tuned.pkl",
                    cache_dir=CACHE_DIR
                )
                reg_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename="models_regressor/sector_income_randomforestmodel.pkl",
                    cache_dir=CACHE_DIR
                )
                
                # Load with memory optimization
                st.write("Loading classifier...")
                clf_data = joblib.load(clf_path, mmap_mode='r' if os.path.getsize(clf_path) > 100*1024*1024 else None)
                st.write("Loading regressor...")
                reg_data = joblib.load(reg_path, mmap_mode='r' if os.path.getsize(reg_path) > 100*1024*1024 else None)
                
                self._models = (clf_data, reg_data)
                gc.collect()  # Force garbage collection
                return self._models
        except Exception as e:
            st.error(f"❌ Failed to load models: {e}")
            import traceback
            st.write(traceback.format_exc())
            return None, None

@st.cache_resource
def load_models():
    """Wrapper for model loading with caching."""
    loader = ModelLoader()
    return loader.get_models()

# ==================== INITIALIZE ====================
st.title("🏠 MPCE Household Prediction")
st.markdown("Predict Monthly Per Capita Expenditure (MPCE) based on household characteristics")

# Load models with error handling
clf_data, reg_data = load_models()

if clf_data is None or reg_data is None:
    st.error("⚠️ Could not load models. This might be a temporary issue. Please refresh the page.")
    st.info("If the issue persists, the Streamlit Cloud instance may need more resources.")
    st.stop()

# Parse models - with defensive programming
try:
    regressors_raw = reg_data.get("models", {})
    if not regressors_raw:
        raise ValueError("No 'models' key in regressor data")
    
    regressors = {}
    for k, v in regressors_raw.items():
        regressors[int(k)] = v
    
    feature_info = reg_data.get("feature_info", {})
    cat_cols = feature_info.get("categorical_cols", [])
    num_cols = feature_info.get("numerical_cols", [])
    encoders = feature_info.get("encoders", {})
    scaler = feature_info.get("scaler")
    
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error parsing models: {e}")
    import traceback
    st.write(traceback.format_exc())
    st.stop()

# ==================== PREPROCESSING ====================
def preprocess_input(input_data_dict):
    """Preprocess input data for prediction."""
    try:
        input_df = pd.DataFrame([input_data_dict])
        encoded_parts = []
        
        # Encode categorical features
        for col in cat_cols:
            if col in input_df.columns and col in encoders:
                encoded = encoders[col].transform(input_df[[col]])
                encoded_parts.append(pd.DataFrame(encoded, columns=[col]))
        
        # Scale numerical features
        if num_cols and scaler:
            scaled = scaler.transform(input_df[num_cols])
            encoded_parts.append(pd.DataFrame(scaled, columns=num_cols))
        
        if encoded_parts:
            result = pd.concat(encoded_parts, axis=1)
            return result
        
        return input_df
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

# ==================== UI ====================
st.markdown("---")
st.write("Enter household details to predict Monthly Per Capita Expenditure (MPCE)")

col1, col2 = st.columns(2)

with col1:
    sector = st.selectbox("Sector", [1, 2], format_func=lambda x: "Rural" if x == 1 else "Urban", key="sector")
    state = st.number_input("State", min_value=1, value=1, key="state")
    nss_region = st.number_input("NSS Region", min_value=1, value=1, key="nss_region")
    district = st.number_input("District", min_value=1, value=1, key="district")
    household_type = st.number_input("Household Type", min_value=1, value=1, key="hh_type")
    religion = st.number_input("Religion", min_value=1, value=1, key="religion")
    social_group = st.number_input("Social Group", min_value=1, value=1, key="social_group")
    hh_size = st.number_input("HH Size", min_value=1, value=4, key="hh_size")

with col2:
    person_count = st.number_input("Person Count", min_value=1, value=4, key="person_count")
    avg_age = st.number_input("Avg Age", min_value=0.0, value=35.0, key="avg_age")
    max_age = st.number_input("Max Age", min_value=0, value=70, key="max_age")
    min_age = st.number_input("Min Age", min_value=0, value=5, key="min_age")
    gender_1_count = st.number_input("Gender 1", min_value=0, value=2, key="g1")
    gender_2_count = st.number_input("Gender 2", min_value=0, value=2, key="g2")
    gender_3_count = st.number_input("Gender 3", min_value=0, value=0, key="g3")
    nco_3d = st.number_input("NCO_3D", min_value=0, value=0, key="nco")

nic_5d = st.number_input("NIC_5D", min_value=0, value=0, key="nic")
avg_education = st.number_input("Avg Education", min_value=0.0, value=8.0, key="avg_ed")
max_education = st.number_input("Max Education", min_value=0.0, value=12.0, key="max_ed")

meals_day_sum = st.number_input("Meals/Day Sum", min_value=0.0, value=3.0, key="meals_day_sum")
meals_day_mean = st.number_input("Meals/Day Mean", min_value=0.0, value=3.0, key="meals_day_mean")
meals_school_sum = st.number_input("School Meals Sum", min_value=0.0, value=0.0, key="meals_school_sum")
meals_school_mean = st.number_input("School Meals Mean", min_value=0.0, value=0.0, key="meals_school_mean")
meals_employer_sum = st.number_input("Employer Meals Sum", min_value=0.0, value=0.0, key="meals_employer_sum")
meals_employer_mean = st.number_input("Employer Meals Mean", min_value=0.0, value=0.0, key="meals_employer_mean")
meals_payment_sum = st.number_input("Paid Meals Sum", min_value=0.0, value=0.0, key="meals_payment_sum")
meals_payment_mean = st.number_input("Paid Meals Mean", min_value=0.0, value=0.0, key="meals_payment_mean")
meals_home_sum = st.number_input("Home Meals Sum", min_value=0.0, value=3.0, key="meals_home_sum")
meals_home_mean = st.number_input("Home Meals Mean", min_value=0.0, value=3.0, key="meals_home_mean")
internet_users_count = st.number_input("Internet Users", min_value=0, value=0, key="internet")

col3, col4, col5 = st.columns(3)

with col3:
    is_online_clothing = st.checkbox("Clothing", value=False, key="c_cloth")
    is_online_footwear = st.checkbox("Footwear", value=False, key="c_foot")
    is_online_furniture = st.checkbox("Furniture", value=False, key="c_furn")
    is_online_mobile = st.checkbox("Mobile Online", value=False, key="c_mob")

with col4:
    is_online_personal = st.checkbox("Personal", value=False, key="c_pers")
    is_online_recreation = st.checkbox("Recreation", value=False, key="c_rec")
    is_online_appliances = st.checkbox("Appliances", value=False, key="c_app")
    is_online_crockery = st.checkbox("Crockery", value=False, key="c_croc")

with col5:
    is_online_sports = st.checkbox("Sports", value=False, key="c_sport")
    is_online_medical = st.checkbox("Medical", value=False, key="c_med")
    is_online_bedding = st.checkbox("Bedding", value=False, key="c_bed")

col6, col7, col8 = st.columns(3)

with col6:
    is_tv = st.checkbox("TV", value=False, key="a_tv")
    is_radio = st.checkbox("Radio", value=False, key="a_radio")
    is_laptop = st.checkbox("Laptop", value=False, key="a_lap")
    is_mobile_handset = st.checkbox("Mobile Asset", value=False, key="a_mob")

with col7:
    is_bicycle = st.checkbox("Bicycle", value=False, key="a_bike")
    is_motorcycle = st.checkbox("Motorcycle", value=False, key="a_moto")
    is_motorcar = st.checkbox("Motor Car", value=False, key="a_car")
    is_trucks = st.checkbox("Trucks", value=False, key="a_truck")

with col8:
    is_animal_cart = st.checkbox("Animal Cart", value=False, key="a_cart")
    is_refrigerator = st.checkbox("Refrigerator", value=False, key="a_ref")
    is_washing_machine = st.checkbox("Washing Machine", value=False, key="a_wash")
    is_ac = st.checkbox("AC/Cooler", value=False, key="a_ac")

# ==================== PREDICTION ====================
if st.button("🔮 Predict MPCE", use_container_width=True, key="predict_btn"):
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
        
        processed_df = preprocess_input(input_data)
        
        if processed_df is None:
            st.error("Could not process input data")
        else:
            if sector not in regressors:
                st.error(f"No model available for sector {sector}")
            else:
                with st.spinner("Making prediction..."):
                    regressor = regressors[sector]
                    predicted_mpce = regressor.predict(processed_df)[0]
                    sector_name = "Rural" if sector == 1 else "Urban"
                    
                    st.success("✅ Prediction Complete!")
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.metric("Predicted MPCE (₹)", f"{predicted_mpce:,.2f}")
                    with col_r2:
                        st.metric("Sector", sector_name)
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.write(traceback.format_exc())
