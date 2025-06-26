from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import List, Dict, Any

# ------------------ Load Models & Preprocessing Info ------------------
# Expect 'sector_income_randomforestmodel.pkl' to contain:
# {"models": {"1": model1, "2": model2, "3": model3, "4": model4}, "feature_info": {...}}
reg_data = joblib.load("models_regressor/sector_income_randomforestmodel.pkl")
# Convert model keys to ints in case they're strings
regressors_raw: Dict[Any, Any] = reg_data["models"]
regressors: Dict[int, Any] = {int(k): v for k, v in regressors_raw.items()}
feature_info: Dict[str, Any] = reg_data["feature_info"]

# Load classifier models (e.g., {'clf_rural': ..., 'clf_urban': ...})
clf_data = joblib.load("models_clf/sector_income_classifiers_tuned.pkl")
classifiers: Dict[str, Any] = clf_data

# Unpack preprocessing assets
cat_cols = feature_info["categorical_cols"]
num_cols = feature_info["numerical_cols"]
encoders = feature_info["encoders"]
scaler = feature_info["scaler"]

# Build expected feature order exactly as in training
def get_expected_feature_order():
    feature_list = []
    feature_list.extend(num_cols)
    for col in cat_cols:
        cats = encoders[col].categories_[0]
        feature_list.extend([f"{col}_{cat}" for cat in cats])
    return feature_list

final_features = get_expected_feature_order()


# ------------------ Utility: Feature Preprocessing ------------------
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
        encoded.append(scaler.transform(raw_df[numerical_cols]))

    return pd.DataFrame(
        pd.np.hstack(encoded)
    )

# ------------------ FastAPI App ------------------
app = FastAPI(
    title="Sector & State MPCE Prediction API",
    version="1.4.0",
    description="Household, state, and state+sector MPCE estimation with preprocessing"
)

# Mapping from Pydantic attribute names to DataFrame column names
attribute_to_column = {
    "Sector": "Sector",
    "State": "State",
    "NSS_Region": "NSS-Region",
    "District": "District",
    "Household_Type": "Household Type",
    "Religion": "Religion of the head of the household",
    "Social_Group": "Social Group of the head of the household",
    "HH_Size": "HH Size (For FDQ)",
    "NCO_3D": "NCO_3D",
    "NIC_5D": "NIC_5D",
    "Is_online_Clothing_Purchased_Last365": "Is_online_Clothing_Purchased_Last365",
    "Is_online_Footwear_Purchased_Last365": "Is_online_Footwear_Purchased_Last365",
    "Is_online_Furniture_fixturesPurchased_Last365": "Is_online_Furniture_fixturesPurchased_Last365",
    "Is_online_Mobile_Handset_Purchased_Last365": "Is_online_Mobile_Handset_Purchased_Last365",
    "Is_online_Personal_Goods_Purchased_Last365": "Is_online_Personal_Goods_Purchased_Last365",
    "Is_online_Recreation_Goods_Purchased_Last365": "Is_online_Recreation_Goods_Purchased_Last365",
    "Is_online_Household_Appliances_Purchased_Last365": "Is_online_Household_Appliances_Purchased_Last365",
    "Is_online_Crockery_Utensils_Purchased_Last365": "Is_online_Crockery_Utensils_Purchased_Last365",
    "Is_online_Sports_Goods_Purchased_Last365": "Is_online_Sports_Goods_Purchased_Last365",
    "Is_online_Medical_Equipment_Purchased_Last365": "Is_online_Medical_Equipment_Purchased_Last365",
    "Is_online_Bedding_Purchased_Last365": "Is_online_Bedding_Purchased_Last365",
    "Is_HH_Have_Television": "Is_HH_Have_Television",
    "Is_HH_Have_Radio": "Is_HH_Have_Radio",
    "Is_HH_Have_Laptop_PC": "Is_HH_Have_Laptop_PC",
    "Is_HH_Have_Mobile_handset": "Is_HH_Have_Mobile_handset",
    "Is_HH_Have_Bicycle": "Is_HH_Have_Bicycle",
    "Is_HH_Have_Motorcycle_scooter": "Is_HH_Have_Motorcycle_scooter",
    "Is_HH_Have_Motorcar_jeep_van": "Is_HH_Have_Motorcar_jeep_van",
    "Is_HH_Have_Trucks": "Is_HH_Have_Trucks",
    "Is_HH_Have_Animal_cart": "Is_HH_Have_Animal_cart",
    "Is_HH_Have_Refrigerator": "Is_HH_Have_Refrigerator",
    "Is_HH_Have_Washing_machine": "Is_HH_Have_Washing_machine",
    "Is_HH_Have_Airconditioner_aircooler": "Is_HH_Have_Airconditioner_aircooler",
    "person_count": "person_count",
    "avg_age": "avg_age",
    "max_age": "max_age",
    "min_age": "min_age",
    "gender_1_count": "gender_1_count",
    "gender_2_count": "gender_2_count",
    "gender_3_count": "gender_3_count",
    "avg_education": "avg_education",
    "max_education": "max_education",
    "No_of_meals_usually_taken_in_a_day_sum": "No. of meals usually taken in a day_sum",
    "No_of_meals_usually_taken_in_a_day_mean": "No. of meals usually taken in a day_mean",
    "No_of_meals_taken_last_30_days_school_sum": "No. of meals taken during last 30 days from school, balwadi etc._sum",
    "No_of_meals_taken_last_30_days_school_mean": "No. of meals taken during last 30 days from school, balwadi etc._mean",
    "No_of_meals_taken_last_30_days_employer_sum": "No. of meals taken during last 30 days from employer as perquisites or part of wage_sum",
    "No_of_meals_taken_last_30_days_employer_mean": "No. of meals taken during last 30 days from employer as perquisites or part of wage_mean",
    "No_of_meals_taken_last_30_days_payment_sum": "No. of meals taken during last 30 days on payment_sum",
    "No_of_meals_taken_last_30_days_payment_mean": "No. of meals taken during last 30 days on payment_mean",
    "No_of_meals_taken_last_30_days_home_sum": "No. of meals taken during last 30 days at home_sum",
    "No_of_meals_taken_last_30_days_home_mean": "No. of meals taken during last 30 days at home_mean",
    "internet_users_count": "internet_users_count"
}

class HouseholdFeatures(BaseModel):
    Sector: int
    State: int
    NSS_Region: int = Field(..., alias="NSS-Region")
    District: int
    Household_Type: int = Field(..., alias="Household Type")
    Religion: int = Field(..., alias="Religion of the head of the household")
    Social_Group: int = Field(..., alias="Social Group of the head of the household")
    HH_Size: int = Field(..., alias="HH Size (For FDQ)")
    NCO_3D: int
    NIC_5D: int
    Is_online_Clothing_Purchased_Last365: int
    Is_online_Footwear_Purchased_Last365: int
    Is_online_Furniture_fixturesPurchased_Last365: int
    Is_online_Mobile_Handset_Purchased_Last365: int
    Is_online_Personal_Goods_Purchased_Last365: int
    Is_online_Recreation_Goods_Purchased_Last365: int
    Is_online_Household_Appliances_Purchased_Last365: int
    Is_online_Crockery_Utensils_Purchased_Last365: int
    Is_online_Sports_Goods_Purchased_Last365: int
    Is_online_Medical_Equipment_Purchased_Last365: int
    Is_online_Bedding_Purchased_Last365: int
    Is_HH_Have_Television: int
    Is_HH_Have_Radio: int
    Is_HH_Have_Laptop_PC: int
    Is_HH_Have_Mobile_handset: int
    Is_HH_Have_Bicycle: int
    Is_HH_Have_Motorcycle_scooter: int
    Is_HH_Have_Motorcar_jeep_van: int
    Is_HH_Have_Trucks: int
    Is_HH_Have_Animal_cart: int
    Is_HH_Have_Refrigerator: int
    Is_HH_Have_Washing_machine: int
    Is_HH_Have_Airconditioner_aircooler: int
    person_count: int
    avg_age: float
    max_age: int
    min_age: int
    gender_1_count: int
    gender_2_count: int
    gender_3_count: int
    avg_education: float
    max_education: float
    No_of_meals_usually_taken_in_a_day_sum: float = Field(..., alias="No. of meals usually taken in a day_sum")
    No_of_meals_usually_taken_in_a_day_mean: float = Field(..., alias="No. of meals usually taken in a day_mean")
    No_of_meals_taken_last_30_days_school_sum: float = Field(..., alias="No. of meals taken during last 30 days from school, balwadi etc._sum")
    No_of_meals_taken_last_30_days_school_mean: float = Field(..., alias="No. of meals taken during last 30 days from school, balwadi etc._mean")
    No_of_meals_taken_last_30_days_employer_sum: float = Field(..., alias="No. of meals taken during last 30 days from employer as perquisites or part of wage_sum")
    No_of_meals_taken_last_30_days_employer_mean: float = Field(..., alias="No. of meals taken during last 30 days from employer as perquisites or part of wage_mean")
    No_of_meals_taken_last_30_days_payment_sum: float = Field(..., alias="No. of meals taken during last 30 days on payment_sum")
    No_of_meals_taken_last_30_days_payment_mean: float = Field(..., alias="No. of meals taken during last 30 days on payment_mean")
    No_of_meals_taken_last_30_days_home_sum: float = Field(..., alias="No. of meals taken during last 30 days at home_sum")
    No_of_meals_taken_last_30_days_home_mean: float = Field(..., alias="No. of meals taken during last 30 days at home_mean")
    internet_users_count: int

    # Add other features as needed...
# ------------------ API Endpoints ------------------
# Health Check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Predict household MPCE
@app.post("/predict/household")
def predict_household(features: HouseholdFeatures):
    # Use attribute names to get values, but DataFrame columns as in training
    data = {v: getattr(features, k) for k, v in attribute_to_column.items()}
    df = pd.DataFrame([data])
    try:
        X = preprocess_features(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")
    sector_val = int(df['Sector'].iloc[0])
    if sector_val == 1:
        clf = classifiers['clf_rural']
        class_pred = clf.predict(X)[0]
        class_id = 2 if class_pred == 1 else 1
        sector_label = "Rural Upper" if class_id == 2 else "Rural Lower"
    elif sector_val == 2:
        clf = classifiers['clf_urban']
        class_pred = clf.predict(X)[0]
        class_id = 4 if class_pred == 1 else 3
        sector_label = "Urban Upper" if class_id == 4 else "Urban Lower"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown sector value: {sector_val}")
    reg = regressors.get(class_id)
    if reg is None:
        raise HTTPException(status_code=500, detail=f"Regressor not found for class_id: {class_id}")
    try:
        pred = reg.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return {
        "State": int(df['State'].iloc[0]),
        "Sector": sector_label,
        "predicted_mpce": float(pred)
    }

# # Compute State MPCE
# @app.post("/compute/state_mpce")
# def compute_state_mpce(req: FilePathRequest):
#     df = pd.read_csv(req.predictions_csv)
#     # unify predicted column
#     if 'predicted_mpce' in df.columns and 'Predicted_MPCE' not in df.columns:
#         df['Predicted_MPCE'] = df['predicted_mpce']
#     weight_cols = [c for c in df.columns if 'weight' in c.lower()]
#     if not weight_cols:
#         raise HTTPException(status_code=400, detail="No weight column found")
#     w = weight_cols[0]
#     test = pd.read_csv(req.test_csv)[['HH_ID','HH Size (For FDQ)']]
#     df = df.merge(test, on='HH_ID')
#     df['wxs'] = df[w]*df['HH Size (For FDQ)']
#     agg = (
#         df.groupby('State')
#           .apply(lambda g: (g.Predicted_MPCE*g.wxs).sum()/g.wxs.sum())
#           .reset_index(name='Predicted State MPCE')
#     )
#     return agg.to_dict(orient='records')

# # Compute State-Sector MPCE
# @app.post("/compute/state_sector_mpce")
# def compute_state_sector_mpce(req: FilePathRequest):
#     df = pd.read_csv(req.predictions_csv)
#     if 'predicted_mpce' in df.columns and 'Predicted_MPCE' not in df.columns:
#         df['Predicted_MPCE'] = df['predicted_mpce']
#     weight_cols = [c for c in df.columns if 'weight' in c.lower()]
#     if not weight_cols:
#         raise HTTPException(status_code=400, detail="No weight column found")
#     w = weight_cols[0]
#     test = pd.read_csv(req.test_csv)[['HH_ID','HH Size (For FDQ)']]
#     df = df.merge(test, on='HH_ID')
#     df['wxs']=df[w]*df['HH Size (For FDQ)']
#     agg = (
#         df.groupby(['State','Sector'])
#           .apply(lambda g: (g.Predicted_MPCE*g.wxs).sum()/g.wxs.sum())
#           .reset_index(name='Predicted State-Sector MPCE')
#     )
#     return agg.to_dict(orient='records')

# To run:
# pip install fastapi uvicorn pandas joblib scikit-learn
# uvicorn app:app --reload
