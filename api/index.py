"""FastAPI app for MPCE predictions."""
import os
import json
import joblib
from typing import Dict, Any
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
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
        print("Using cached models")
        return _models_cache
    
    try:
        print("Loading models from Hugging Face...")
        MODEL_REPO = "shubh-iiit/mpce-models"
        CACHE_DIR = "/tmp/mpce_cache"
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Download regressor only (simpler and sufficient for predictions)
        print(f"Downloading from {MODEL_REPO}")
        reg_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="models_regressor/sector_income_randomforestmodel.pkl",
            cache_dir=CACHE_DIR
        )
        print(f"Loading model from {reg_path}")
        reg_data = joblib.load(reg_path)
        print("Model loaded successfully")

        _models_cache = (None, reg_data)  # No classifier needed for now
        return _models_cache
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Failed to load models: {str(e)}")@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "MPCE Prediction API"}

@app.get("/")
async def root():
    """Serve the main HTML page."""
    static_dir = Path(__file__).parent.parent / "public"
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
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

# Mount static files for CSS/JS/images (but not root)
static_dir = Path(__file__).parent.parent / "public"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
