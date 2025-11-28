"""FastAPI app for MPCE predictions."""
import os
import json
import joblib
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
import pandas as pd

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_models_cache = None

def load_models():
    """Load models from Hugging Face Hub."""
    global _models_cache
    
    if _models_cache is not None:
        return _models_cache
    
    try:
        MODEL_REPO = "shubh-iiit/mpce-models"
        CACHE_DIR = "/tmp/mpce_cache"
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Download classifier
        clf_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="models_clf/sector_income_classifiers_tuned.pkl",
            cache_dir=CACHE_DIR
        )
        clf_data = joblib.load(clf_path)
        
        # Download regressor
        reg_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="models_regressor/sector_income_randomforestmodel.pkl",
            cache_dir=CACHE_DIR
        )
        reg_data = joblib.load(reg_path)
        
        _models_cache = (clf_data, reg_data)
        return _models_cache
    except Exception as e:
        raise Exception(f"Failed to load models: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "MPCE Prediction API"}

@app.post("/api/predict")
async def predict(data: Dict[str, Any]):
    """Make a prediction."""
    try:
        # Load models
        clf_data, reg_data = load_models()
        
        # Parse models
        regressors_raw = reg_data.get("models", {})
        regressors = {int(k): v for k, v in regressors_raw.items()}
        feature_info = reg_data.get("feature_info", {})
        
        cat_cols = feature_info.get("categorical_cols", [])
        num_cols = feature_info.get("numerical_cols", [])
        encoders = feature_info.get("encoders", {})
        scaler = feature_info.get("scaler")
        
        # Extract input
        sector = int(data.get("sector", 1))
        input_df = pd.DataFrame([data])
        
        # Preprocess
        encoded_parts = []
        for col in cat_cols:
            if col in input_df.columns and col in encoders:
                encoded = encoders[col].transform(input_df[[col]])
                encoded_parts.append(pd.DataFrame(encoded, columns=[col]))
        
        if num_cols and scaler:
            scaled = scaler.transform(input_df[num_cols])
            encoded_parts.append(pd.DataFrame(scaled, columns=num_cols))
        
        processed_df = pd.concat(encoded_parts, axis=1) if encoded_parts else input_df
        
        # Predict
        if sector not in regressors:
            return JSONResponse(
                {"error": f"No model for sector {sector}"},
                status_code=400
            )
        
        regressor = regressors[sector]
        predicted_mpce = float(regressor.predict(processed_df)[0])
        sector_name = "Rural" if sector == 1 else "Urban"
        
        return {
            "success": True,
            "mpce": predicted_mpce,
            "sector": sector_name,
            "sector_code": sector
        }
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

# Serve static files
try:
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
except:
    pass
