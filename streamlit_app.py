# app_streamlit_parfum.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
# --- SafeSVD definition (must appear before loading saved pipeline) ---
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin

class SafeSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=50, random_state=42):
        self.n_components = int(n_components)
        self.random_state = random_state
        self.svd_ = None
        self.do_reduce_ = False
    def fit(self, X, y=None):
        try:
            n_features = X.shape[1]
        except Exception:
            X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            n_features = X_arr.shape[1]
        if n_features is None:
            self.do_reduce_ = False
            return self
        if n_features > 1 and self.n_components < n_features:
            n_comp = min(self.n_components, n_features - 1)
            if n_comp >= 1:
                self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
                self.svd_.fit(X)
                self.do_reduce_ = True
            else:
                self.do_reduce_ = False
        else:
            self.do_reduce_ = False
        return self
    def transform(self, X):
        if self.do_reduce_ and self.svd_ is not None:
            return self.svd_.transform(X)
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

st.set_page_config(page_title="Perfume Situation Classifier", layout="centered")

# -----------------------
# Utilities: preprocessing same as training
# -----------------------
def normalize_text(s: str) -> str:
    s = '' if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r'[\/;|•]+', ',', s)
    s = re.sub(r'\s*[,\;]\s*', ', ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def build_single_df(top_notes, mid_notes, base_notes, brand, concentrate, gender, price, size):
    df = pd.DataFrame([{
        'top notes': top_notes,
        'mid notes': mid_notes,
        'base notes': base_notes,
        'brand': brand,
        'concentrate': concentrate,
        'gender': gender,
        'price': price,
        'size': size
    }])
    # normalize fields
    text_cols = ['top notes','mid notes','base notes','brand','concentrate','gender']
    for c in text_cols:
        df[c] = df[c].fillna('').astype(str).apply(normalize_text)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    df['size']  = pd.to_numeric(df['size'], errors='coerce').fillna(0.0)
    df['all_notes'] = (df['top notes'].fillna('') + ', ' + df['mid notes'].fillna('') + ', ' + df['base notes'].fillna(''))
    df['all_notes'] = df['all_notes'].str.replace(r'\s+', ' ', regex=True).str.strip(', ')
    return df

# -----------------------
# Sidebar: model selection / upload
# -----------------------
st.sidebar.header("Model & Data")
default_model_path = "full_pipeline_situation_model.joblib"
model_path = st.sidebar.text_input("Model path (.joblib)", value=default_model_path)
uploaded_model = st.sidebar.file_uploader("Or upload pipeline .joblib", type=["joblib","pkl"])
use_prob = st.sidebar.checkbox("Show prediction probabilities (if available)", value=True)

# load model
model = None
if uploaded_model is not None:
    try:
        model = joblib.load(uploaded_model)
        st.sidebar.success("Model loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")
else:
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.sidebar.success(f"Model loaded: {model_path}")
        except Exception as e:
            st.sidebar.error(f"Failed to load model from path: {e}")
    else:
        st.sidebar.warning("No model file found. Provide valid path or upload .joblib file.")

st.title("Perfume Situation Classifier — Day / Night / Versatile")
st.markdown(
    """
    Aplikasi sederhana untuk memprediksi class *situation* parfum (day / night / versatile)
    berdasarkan notes, brand, concentrate, gender, price, dan size.
    \n**Input**: bisa dimasukkan manual per-item atau upload file CSV (batch).
    """
)

# -----------------------
# Main: Manual single prediction
# -----------------------
st.header("Single prediction")

col1, col2 = st.columns(2)
with col1:
    perfume_name = st.text_input("Perfume name (optional)", value="")
    top_notes = st.text_area("Top notes (comma-separated)", value="", height=80)
    mid_notes = st.text_area("Mid notes (comma-separated)", value="", height=80)
    base_notes = st.text_area("Base notes (comma-separated)", value="", height=80)

with col2:
    brand = st.text_input("Brand", value="")
    concentrate = st.text_input("Concentrate (e.g., edp, xdp)", value="")
    gender = st.selectbox("Gender", options=["unisex","female","male",""], index=0)
    price = st.number_input("Price (numeric)", min_value=0.0, value=0.0, step=1000.0)
    size = st.number_input("Size (ml)", min_value=0, value=50, step=1)

predict_button = st.button("Predict single item")

if predict_button:
    sample_df = build_single_df(top_notes, mid_notes, base_notes, brand, concentrate, gender, price, size)
    if model is None:
        st.error("Model tidak terload. Upload atau masukkan path model (.joblib) di sidebar.")
    else:
        try:
            # Many pipelines expect DataFrame with 'all_notes' + categorical + numeric order.
            # Here we pass columns used in training: all_notes, brand, concentrate, gender, price, size
            X_input = sample_df[['all_notes','brand','concentrate','gender','price','size']]
            pred = model.predict(X_input)
            st.success(f"Predicted situation: **{pred[0]}**")
            if use_prob and hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[0]
                classes = model.classes_
                prob_df = pd.DataFrame({"class": classes, "probability": probs})
                prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
                st.table(prob_df)
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")

# -----------------------
# Batch prediction via CSV
# -----------------------
st.header("Batch prediction (CSV)")

st.markdown("Unggah CSV berisi kolom: `top notes`, `mid notes`, `base notes`, `brand`, `concentrate`, `gender`, `price`, `size`")
uploaded_csv = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
if uploaded_csv is not None:
    try:
        df_batch = pd.read_csv(uploaded_csv)
        st.write("Preview uploaded data:")
        st.dataframe(df_batch.head())
        # prepare
        required_cols = ['top notes', 'mid notes', 'base notes', 'brand', 'concentrate', 'gender', 'price', 'size']
        missing = [c for c in required_cols if c not in df_batch.columns]
        if missing:
            st.error(f"CSV missing required columns: {missing}")
        else:
            # normalize and create all_notes
            for c in ['top notes','mid notes','base notes','brand','concentrate','gender']:
                df_batch[c] = df_batch[c].fillna('').astype(str).apply(normalize_text)
            df_batch['price'] = pd.to_numeric(df_batch['price'], errors='coerce').fillna(0.0)
            df_batch['size']  = pd.to_numeric(df_batch['size'], errors='coerce').fillna(0.0)
            df_batch['all_notes'] = (df_batch['top notes'] + ', ' + df_batch['mid notes'] + ', ' + df_batch['base notes']).str.replace(r'\s+', ' ', regex=True).str.strip(', ')

            if model is None:
                st.error("Model tidak terload. Upload atau masukkan path model (.joblib) di sidebar.")
            else:
                try:
                    X_in = df_batch[['all_notes','brand','concentrate','gender','price','size']]
                    preds = model.predict(X_in)
                    df_batch['predicted_situation'] = preds
                    # show probs if available
                    if use_prob and hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_in)
                        # attach top probability
                        df_batch['pred_prob_max'] = np.max(probs, axis=1)
                        # optionally attach per-class probs by column
                        for i, cls in enumerate(model.classes_):
                            df_batch[f"prob_{cls}"] = probs[:, i]
                    st.success("Batch prediction selesai.")
                    st.dataframe(df_batch.head(20))
                    # offer download
                    csv_out = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button("Download results CSV", csv_out, file_name="predictions_parfum.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Error saat prediksi batch: {e}")
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")

# -----------------------
# Footer / help
# -----------------------
st.markdown("---")
"""
st.markdown(
    
    **Notes**:
    - Pastikan model pipeline yang Anda muat dibuat dengan preprocessing yang **sama**:
      kolom `all_notes`, `brand`, `concentrate`, `gender`, `price`, `size` digunakan saat training.
    - Jika pelatihan dilakukan dengan SMOTE (imblearn) pipeline, pastikan `imbalanced-learn` terinstal di environment.
    - Jika model tidak ter-load, Anda dapat melatih ulang pipeline menggunakan skrip `train_full_pipeline_parfum_with_imbalance.py` yang sudah diberikan sebelumnya.
    
)
"""