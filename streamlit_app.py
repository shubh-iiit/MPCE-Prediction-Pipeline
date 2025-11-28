import streamlit as st

st.set_page_config(page_title="MPCE Pipeline Demo", layout="wide")
st.title("MPCE Household Prediction Pipeline")

st.markdown("""
## Project Overview
This is an ML pipeline for predicting Monthly Per Capita Expenditure (MPCE) 
of Indian households using classification and regression models.

### Key Resources
- **[Full Analysis Notebook](https://github.com/shubh-iiit/MPCE-Prediction-Pipeline/blob/main/Insights.ipynb)** - Detailed EDA and model training
- **[GitHub Repository](https://github.com/shubh-iiit/MPCE-Prediction-Pipeline)** - Source code and documentation
- **[Landing Page](https://shubh-iiit.github.io/MPCE-Prediction-Pipeline/)** - Project overview

### Model Details
- **Classification**: Sector classification (Rural/Urban)
- **Regression**: MPCE prediction per household
- **Algorithms**: XGBoost, Random Forest
- **Features**: 30+ household characteristics

### Data
- Source: NSS (National Sample Survey) - MPCE Data
- Target: Monthly household expenditure prediction
- Split: Training and testing datasets included

**Status**: ✅ Production Ready | **Deployment**: GitHub Pages + Streamlit Cloud
""")

st.info("📌 For full interactive predictions with models, visit the live demo link above.")
