import streamlit as st
import requests

st.title("MPCE Household Prediction")

st.write("Enter household details below to get MPCE prediction.")

# Helper for binary dropdowns
def binary_dropdown(label):
    return st.selectbox(label, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Main features
sector = st.selectbox("Sector", [1, 2], format_func=lambda x: "Rural" if x == 1 else "Urban")
state = st.number_input("State (as integer code)", min_value=1)
nss_region = st.number_input("NSS-Region", min_value=1)
district = st.number_input("District", min_value=1)
household_type = st.number_input("Household Type (as integer code)", min_value=1)
religion = st.number_input("Religion of the head of the household (as integer code)", min_value=1)
social_group = st.number_input("Social Group of the head of the household (as integer code)", min_value=1)
hh_size = st.number_input("HH Size (For FDQ)", min_value=1)
nco_3d = st.number_input("NCO_3D", min_value=0)
nic_5d = st.number_input("NIC_5D", min_value=0)

# Binary features (dropdowns)
is_online_clothing = binary_dropdown("Is_online_Clothing_Purchased_Last365")
is_online_footwear = binary_dropdown("Is_online_Footwear_Purchased_Last365")
is_online_furniture = binary_dropdown("Is_online_Furniture_fixturesPurchased_Last365")
is_online_mobile = binary_dropdown("Is_online_Mobile_Handset_Purchased_Last365")
is_online_personal = binary_dropdown("Is_online_Personal_Goods_Purchased_Last365")
is_online_recreation = binary_dropdown("Is_online_Recreation_Goods_Purchased_Last365")
is_online_appliances = binary_dropdown("Is_online_Household_Appliances_Purchased_Last365")
is_online_crockery = binary_dropdown("Is_online_Crockery_Utensils_Purchased_Last365")
is_online_sports = binary_dropdown("Is_online_Sports_Goods_Purchased_Last365")
is_online_medical = binary_dropdown("Is_online_Medical_Equipment_Purchased_Last365")
is_online_bedding = binary_dropdown("Is_online_Bedding_Purchased_Last365")
is_tv = binary_dropdown("Is_HH_Have_Television")
is_radio = binary_dropdown("Is_HH_Have_Radio")
is_laptop = binary_dropdown("Is_HH_Have_Laptop_PC")
is_mobile_handset = binary_dropdown("Is_HH_Have_Mobile_handset")
is_bicycle = binary_dropdown("Is_HH_Have_Bicycle")
is_motorcycle = binary_dropdown("Is_HH_Have_Motorcycle_scooter")
is_motorcar = binary_dropdown("Is_HH_Have_Motorcar_jeep_van")
is_trucks = binary_dropdown("Is_HH_Have_Trucks")
is_animal_cart = binary_dropdown("Is_HH_Have_Animal_cart")
is_refrigerator = binary_dropdown("Is_HH_Have_Refrigerator")
is_washing_machine = binary_dropdown("Is_HH_Have_Washing_machine")
is_ac = binary_dropdown("Is_HH_Have_Airconditioner_aircooler")

# Numeric features
person_count = st.number_input("person_count", min_value=1)
avg_age = st.number_input("avg_age", min_value=0.0)
max_age = st.number_input("max_age", min_value=0)
min_age = st.number_input("min_age", min_value=0)
gender_1_count = st.number_input("gender_1_count", min_value=0)
gender_2_count = st.number_input("gender_2_count", min_value=0)
gender_3_count = st.number_input("gender_3_count", min_value=0)
avg_education = st.number_input("avg_education", min_value=0.0)
max_education = st.number_input("max_education", min_value=0.0)

# Meals features
meals_day_sum = st.number_input("No. of meals usually taken in a day_sum", min_value=0.0)
meals_day_mean = st.number_input("No. of meals usually taken in a day_mean", min_value=0.0)
meals_school_sum = st.number_input("No. of meals taken during last 30 days from school, balwadi etc._sum", min_value=0.0)
meals_school_mean = st.number_input("No. of meals taken during last 30 days from school, balwadi etc._mean", min_value=0.0)
meals_employer_sum = st.number_input("No. of meals taken during last 30 days from employer as perquisites or part of wage_sum", min_value=0.0)
meals_employer_mean = st.number_input("No. of meals taken during last 30 days from employer as perquisites or part of wage_mean", min_value=0.0)
meals_payment_sum = st.number_input("No. of meals taken during last 30 days on payment_sum", min_value=0.0)
meals_payment_mean = st.number_input("No. of meals taken during last 30 days on payment_mean", min_value=0.0)
meals_home_sum = st.number_input("No. of meals taken during last 30 days at home_sum", min_value=0.0)
meals_home_mean = st.number_input("No. of meals taken during last 30 days at home_mean", min_value=0.0)

internet_users_count = st.number_input("internet_users_count", min_value=0)

# Prepare payload
payload = {
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

if st.button("Predict MPCE"):
    try:
        response = requests.post("http://localhost:8000/predict/household", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted MPCE: ₹{result['predicted_mpce']:.2f}")
            st.write("Classified as:", result["Sector"])
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Could not connect to FastAPI backend: {e}")

st.info("Make sure your FastAPI server is running at http://localhost:8000")