# feature_importance_classifiers.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CLASSIFIER_PICKLE = "/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/models_Clf/sector_income_classifiers_tuned.pkl"
REGRESSOR_PICKLE  = "/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/models_Regressor/sector_income_randomforestmodel.pkl"
TRAIN_WITH_CLASS  = "/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/train_with_class.csv"
TOP_K             = 20


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def get_feature_names(feature_info):
    names = []
    for col in feature_info["categorical_cols"]:
        cats = feature_info["encoders"][col].categories_[0]
        names += [f"{col}_{cat}" for cat in cats]
    names += feature_info["numerical_cols"]
    return names

def print_topk_importances(importances, feature_names, model_name, k=20):
    indices = np.argsort(importances)[::-1][:k]
    print(f"\nTop {k} features for {model_name}:\n")
    for rank, idx in enumerate(indices, 1):
        print(f"{rank:2d}. {feature_names[idx]:<40} {importances[idx]:.4f}")


# ─── 1. Load feature_info from your regressor pickle ──────────────────────────
reg_data    = joblib.load(REGRESSOR_PICKLE)
feature_info= reg_data["feature_info"]
feature_names = get_feature_names(feature_info)

# ─── 2. Load the tuned classifiers ────────────────────────────────────────────
clf_data  = joblib.load(CLASSIFIER_PICKLE)
clf_rural = clf_data["clf_rural"]
clf_urban = clf_data["clf_urban"]

# ─── 3. Extract and print importances ────────────────────────────────────────
print_topk_importances(clf_rural.feature_importances_, feature_names, "Rural Classifier", TOP_K)
print_topk_importances(clf_urban.feature_importances_, feature_names, "Urban Classifier", TOP_K)



# ─── HELPERS ───────────────────────────────────────────────────────────────────
def get_feature_names(feature_info):
    names = []
    for col in feature_info["categorical_cols"]:
        cats = feature_info["encoders"][col].categories_[0]
        names += [f"{col}_{cat}" for cat in cats]
    names += feature_info["numerical_cols"]
    return names

def topk_features(importances, feature_names, k=20):
    idxs = np.argsort(importances)[::-1][:k]
    return [feature_names[i] for i in idxs]


# ─── 1. Load feature_info and feature names ───────────────────────────────────
reg_data      = joblib.load(REGRESSOR_PICKLE)
feature_info  = reg_data["feature_info"]
feature_names = get_feature_names(feature_info)

# ─── 2. Load classifiers ──────────────────────────────────────────────────────
clf_data  = joblib.load(CLASSIFIER_PICKLE)
clf_rural = clf_data["clf_rural"]
clf_urban = clf_data["clf_urban"]

# ─── 3. Identify top K features per classifier ────────────────────────────────
top_rural = topk_features(clf_rural.feature_importances_, feature_names, TOP_K)
top_urban = topk_features(clf_urban.feature_importances_, feature_names, TOP_K)

# ─── 4. Load full training set ─────────────────────────────────────────────────
df = pd.read_csv(TRAIN_WITH_CLASS)

# ─── 5. Compute correlations for Rural (classes 1 vs 2) ───────────────────────
rural_df = df[df["Income_Class"].isin([1, 2])].dropna(subset=top_rural)
# Binary target: 1 = upper (class 2), 0 = lower (class 1)
rural_df["is_upper"] = (rural_df["Income_Class"] == 2).astype(int)

rural_corrs = []
for feat in top_rural:
    corr = rural_df[[feat, "is_upper"]].corr().loc[feat, "is_upper"]
    rural_corrs.append((feat, corr))
rural_corrs = sorted(rural_corrs, key=lambda x: abs(x[1]), reverse=True)

print(f"\nRural Classifier Top {TOP_K} Feature Correlation with (Upper Rural=1):\n")
for feat, corr in rural_corrs:
    print(f"{feat:<40s} {corr:.4f}")

# ─── 6. Compute correlations for Urban (classes 3 vs 4) ───────────────────────
urban_df = df[df["Income_Class"].isin([3, 4])].dropna(subset=top_urban)
# Binary target: 1 = upper (class 4), 0 = lower (class 3)
urban_df["is_upper"] = (urban_df["Income_Class"] == 4).astype(int)

urban_corrs = []
for feat in top_urban:
    corr = urban_df[[feat, "is_upper"]].corr().loc[feat, "is_upper"]
    urban_corrs.append((feat, corr))
urban_corrs = sorted(urban_corrs, key=lambda x: abs(x[1]), reverse=True)

print(f"\nUrban Classifier Top {TOP_K} Feature Correlation with (Upper Urban=1):\n")
for feat, corr in urban_corrs:
    print(f"{feat:<40s} {corr:.4f}")


# ─── LOAD IMPORTS & HELPERS ─────────────────────────────────────────────────────
def get_feature_names(feature_info):
    names = []
    for col in feature_info["categorical_cols"]:
        cats = feature_info["encoders"][col].categories_[0]
        names += [f"{col}_{cat}" for cat in cats]
    names += feature_info["numerical_cols"]
    return names

def topk_features(importances, feature_names, k=20):
    idxs = np.argsort(importances)[::-1][:k]
    return [feature_names[i] for i in idxs]


# ─── 1. Load feature_info & feature names ─────────────────────────────────────
reg = joblib.load(REGRESSOR_PICKLE)
feature_info  = reg["feature_info"]
feature_names = get_feature_names(feature_info)

# ─── 2. Load classifiers & pick top features ──────────────────────────────────
clf = joblib.load(CLASSIFIER_PICKLE)
clf_rural = clf["clf_rural"]
clf_urban = clf["clf_urban"]

top_rural = topk_features(clf_rural.feature_importances_, feature_names, TOP_K)
top_urban = topk_features(clf_urban.feature_importances_, feature_names, TOP_K)

# ─── 3. Load data and compute correlations ────────────────────────────────────
df = pd.read_csv(TRAIN_WITH_CLASS)

# Rural correlations
r = df[df["Income_Class"].isin([1,2])].dropna(subset=top_rural).copy()
r["is_upper"] = (r["Income_Class"]==2).astype(int)
r_corrs = {feat: r[[feat,"is_upper"]].corr().loc[feat,"is_upper"] for feat in top_rural}
r_corrs = pd.Series(r_corrs).abs().sort_values(ascending=True)  # sort for barh

# Urban correlations
u = df[df["Income_Class"].isin([3,4])].dropna(subset=top_urban).copy()
u["is_upper"] = (u["Income_Class"]==4).astype(int)
u_corrs = {feat: u[[feat,"is_upper"]].corr().loc[feat,"is_upper"] for feat in top_urban}
u_corrs = pd.Series(u_corrs).abs().sort_values(ascending=True)

# ─── 4. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2, figsize=(12,8), sharex=False)

# Rural plot
axes[0].barh(r_corrs.index, r_corrs.values, color="#4c72b0")
axes[0].set_title("Rural: |corr(feature, is_upper)|")
axes[0].set_xlabel("Absolute Pearson r")
axes[0].tick_params(axis='y', labelsize=8)

# Urban plot
axes[1].barh(u_corrs.index, u_corrs.values, color="#c44e52")
axes[1].set_title("Urban: |corr(feature, is_upper)|")
axes[1].set_xlabel("Absolute Pearson r")
axes[1].tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.show()

# ─── 3. Pick one sector (rural) and assemble DataFrame ────────────────────────
df = pd.read_csv(TRAIN_WITH_CLASS)
r = df[df["Income_Class"].isin([1,2])].copy()
r["is_upper"] = (r["Income_Class"]==2).astype(int)
data = r[top_rural + ["is_upper"]].dropna()

# ─── 4. Compute correlation matrix ─────────────────────────────────────────────
corr = data.corr()

# ─── 5. Plot with Matplotlib ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,10))
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)

# set ticks
labels = corr.columns
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_yticklabels(labels, fontsize=6)

ax.set_title("Correlation Matrix (Rural Top 20 + is_upper)", pad=20)
plt.tight_layout()
plt.show()

# ─── 3. Pick one sector (rural) and assemble DataFrame ────────────────────────
df = pd.read_csv(TRAIN_WITH_CLASS)
r = df[df["Income_Class"].isin([3,4])].copy()
r["is_upper"] = (r["Income_Class"]==4).astype(int)
data = r[top_urban + ["is_upper"]].dropna()

# ─── 4. Compute correlation matrix ─────────────────────────────────────────────
corr = data.corr()

# ─── 5. Plot with Matplotlib ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,10))
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)

# set ticks
labels = corr.columns
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_yticklabels(labels, fontsize=6)

ax.set_title("Correlation Matrix (Urban Top 20 + is_upper)", pad=20)
plt.tight_layout()
plt.show()