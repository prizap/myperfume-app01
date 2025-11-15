# app_parfume_deploy.py
# Streamlit app for deploying pre-built KDD artifacts (no training inside app)
#
# Expects artifact folder structure (default: ./kdd_artifacts or /mnt/data/kdd_artifacts):
# - classifier_pipeline.joblib            (optional, for prediction)
# - label_classes.json OR label_encoder.joblib (optional, for mapping labels)
# - preprocessed_snapshot.csv             (recommended; used for recommender & visual)
# - notes_vectorizer.joblib               (optional; TF-IDF for recommender)
# - notes_sim_matrix.joblib               (optional; similarity matrix)
# - kmeans_model.joblib                   (optional; clustering)
#
# You can also upload artifacts via UI.
#
# Run:
#   streamlit run app_parfume_deploy.py

import streamlit as st
st.set_page_config(layout="wide", page_title="Parfume KDD - Model Deploy")

import os
from pathlib import Path
import json
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# wordcloud optional
try:
    from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

# -------------------------
# Config & helpers
# -------------------------
DEFAULT_ARTIFACT_DIRS = ["./kdd_artifacts", "/mnt/data/kdd_artifacts"]
DEFAULT_SNAPSHOT = "preprocessed_snapshot.csv"
DEFAULT_PIPELINE = "classifier_pipeline.joblib"
DEFAULT_LABELS_JSON = "label_classes.json"
DEFAULT_VEC = "notes_vectorizer.joblib"
DEFAULT_SIM = "notes_sim_matrix.joblib"
DEFAULT_KMEANS = "kmeans_model.joblib"

def find_first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_load_joblib(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load joblib {path}: {e}")
        return None

def clean_text_notes(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = s.replace(";", ",").replace("|", ",")
    import re
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9, ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ", ", s).strip()
    return s

def generate_wordcloud_image(text, max_words=200, width=800, height=400):
    if not HAS_WORDCLOUD or not text or str(text).strip()=="":
        return None
    stopwords = set(WC_STOPWORDS)
    # small Indonesian stoplist
    indo = {"dan","yang","di","ke","dari","untuk","pada","ada","ini","itu","atau","dengan","sebagai"}
    stopwords = stopwords.union(indo)
    wc = WordCloud(width=width, height=height, background_color="white", stopwords=stopwords, max_words=max_words, collocations=False)
    wc.generate(text)
    return wc

# -------------------------
# UI: artifact selection / uploads
# -------------------------
st.title("Parfume KDD — Deployed Model (Streamlit)")

st.sidebar.header("Artifact options")
use_default = st.sidebar.checkbox("Use default artifact folder (search common locations)", value=True)
artifact_dir = None
if use_default:
    cand = find_first_existing(DEFAULT_ARTIFACT_DIRS)
    if cand:
        artifact_dir = cand
        st.sidebar.info(f"Using artifact dir: {artifact_dir}")
    else:
        st.sidebar.warning("No default artifact dir found. You can upload artifacts manually below or specify a folder.")
upload_snapshot = st.sidebar.file_uploader("Upload preprocessed_snapshot.csv (optional)", type=["csv"])
upload_pipeline = st.sidebar.file_uploader("Upload classifier_pipeline.joblib (optional)", type=["joblib"])
upload_labeljson = st.sidebar.file_uploader("Upload label_classes.json (optional)", type=["json"])
upload_label_joblib = st.sidebar.file_uploader("Upload label_encoder.joblib (optional)", type=["joblib"])
upload_vectorizer = st.sidebar.file_uploader("Upload notes_vectorizer.joblib (optional)", type=["joblib"])
upload_sim = st.sidebar.file_uploader("Upload notes_sim_matrix.joblib (optional)", type=["joblib"])
upload_kmeans = st.sidebar.file_uploader("Upload kmeans_model.joblib (optional)", type=["joblib"])

# allow user to type artifact dir manually
user_specified_dir = st.sidebar.text_input("Or enter artifact directory path (optional)", value="")
if user_specified_dir.strip() != "":
    p = Path(user_specified_dir.strip())
    if p.exists():
        artifact_dir = p
        st.sidebar.success(f"Using artifact dir: {artifact_dir}")
    else:
        st.sidebar.error("Specified directory does not exist")

# Determine actual sources
def get_snapshot_df():
    # priority: uploaded file > artifact_dir/preprocessed_snapshot.csv
    if upload_snapshot:
        try:
            df = pd.read_csv(upload_snapshot)
            st.sidebar.success("Loaded uploaded snapshot CSV")
            return df
        except Exception as e:
            st.sidebar.error(f"Failed load uploaded CSV: {e}")
            return None
    # try artifact dir
    if artifact_dir:
        p = Path(artifact_dir) / DEFAULT_SNAPSHOT
        if p.exists():
            try:
                df = pd.read_csv(p)
                st.sidebar.success(f"Loaded snapshot from {p}")
                return df
            except Exception as e:
                st.sidebar.error(f"Failed to load snapshot from {p}: {e}")
    st.sidebar.info("No snapshot available yet. You can upload CSV snapshot.")
    return None

def get_pipeline():
    # priority: uploaded file > artifact_dir/classifier_pipeline.joblib
    if upload_pipeline:
        # save temporary to disk and load
        try:
            tmp = Path(tempfile.gettempdir()) / "uploaded_pipeline.joblib"
            with open(tmp, "wb") as f:
                f.write(upload_pipeline.getbuffer())
            pipe = safe_load_joblib(tmp)
            if pipe is not None:
                st.sidebar.success("Loaded uploaded pipeline")
            return pipe
        except Exception as e:
            st.sidebar.error(f"Failed load uploaded pipeline: {e}")
            return None
    if artifact_dir:
        p = Path(artifact_dir) / DEFAULT_PIPELINE
        if p.exists():
            pipe = safe_load_joblib(p)
            if pipe is not None:
                st.sidebar.success(f"Loaded pipeline from {p}")
            return pipe
    st.sidebar.info("No classifier pipeline loaded (prediction disabled).")
    return None

def get_label_classes(pipeline_loaded):
    # try uploaded json
    if upload_labeljson:
        try:
            tmp = Path(tempfile.gettempdir()) / "uploaded_label_classes.json"
            with open(tmp, "wb") as f:
                f.write(upload_labeljson.getbuffer())
            data = load_json(tmp)
            if data and "classes" in data:
                st.sidebar.success("Loaded uploaded label_classes.json")
                return data["classes"]
        except Exception:
            pass
    # try uploaded label encoder joblib
    if upload_label_joblib:
        try:
            tmp = Path(tempfile.gettempdir()) / "uploaded_label_encoder.joblib"
            with open(tmp, "wb") as f:
                f.write(upload_label_joblib.getbuffer())
            le = safe_load_joblib(tmp)
            if hasattr(le, "classes_"):
                st.sidebar.success("Loaded uploaded label_encoder.joblib")
                return list(le.classes_)
        except Exception:
            pass
    # try artifact dir JSON
    if artifact_dir:
        p = Path(artifact_dir) / DEFAULT_LABELS_JSON
        if p.exists():
            data = load_json(p)
            if data and "classes" in data:
                st.sidebar.success(f"Loaded label classes from {p}")
                return data["classes"]
    # else, try to inspect pipeline (if pipeline_loaded is not None)
    if pipeline_loaded is not None:
        # if pipeline was trained with LabelEncoder serialized elsewhere, we may not find classes.
        # fallback: no labels
        st.sidebar.info("No label classes found in artifacts. Prediction UI will show encoded integers.")
    return None

def get_vectorizer_and_sim():
    # Uploaded vectorizer
    if upload_vectorizer:
        try:
            tmp = Path(tempfile.gettempdir()) / "uploaded_vectorizer.joblib"
            with open(tmp, "wb") as f:
                f.write(upload_vectorizer.getbuffer())
            vect_obj = safe_load_joblib(tmp)
            if isinstance(vect_obj, dict) and "vectorizer" in vect_obj:
                vect = vect_obj["vectorizer"]
            else:
                vect = vect_obj
            st.sidebar.success("Loaded uploaded vectorizer")
        except Exception as e:
            st.sidebar.error(f"Failed load uploaded vectorizer: {e}")
            vect = None
        # sim matrix upload optional
        if upload_sim:
            try:
                tmp2 = Path(tempfile.gettempdir()) / "uploaded_sim.joblib"
                with open(tmp2, "wb") as f:
                    f.write(upload_sim.getbuffer())
                sim_obj = safe_load_joblib(tmp2)
                sim = sim_obj.get("sim_matrix") if isinstance(sim_obj, dict) and "sim_matrix" in sim_obj else sim_obj
                st.sidebar.success("Loaded uploaded sim matrix")
            except Exception as e:
                st.sidebar.warning(f"Failed load uploaded sim matrix: {e}")
                sim = None
        else:
            sim = None
        return vect, sim

    # try artifact dir
    if artifact_dir:
        vpath = Path(artifact_dir) / DEFAULT_VEC
        spath = Path(artifact_dir) / DEFAULT_SIM
        vect = None
        sim = None
        if vpath.exists():
            obj = safe_load_joblib(vpath)
            # if obj is wrapper dict
            if isinstance(obj, dict) and "vectorizer" in obj:
                vect = obj["vectorizer"]
            else:
                vect = obj
            st.sidebar.success(f"Loaded vectorizer {vpath}")
        if spath.exists():
            sobj = safe_load_joblib(spath)
            sim = sobj.get("sim_matrix") if isinstance(sobj, dict) and "sim_matrix" in sobj else sobj
            st.sidebar.success(f"Loaded sim matrix {spath}")
        return vect, sim
    return None, None

def get_kmeans():
    if upload_kmeans:
        try:
            tmp = Path(tempfile.gettempdir()) / "uploaded_kmeans.joblib"
            with open(tmp, "wb") as f:
                f.write(upload_kmeans.getbuffer())
            km = safe_load_joblib(tmp)
            st.sidebar.success("Loaded uploaded kmeans model")
            return km
        except Exception:
            pass
    if artifact_dir:
        p = Path(artifact_dir) / DEFAULT_KMEANS
        if p.exists():
            km = safe_load_joblib(p)
            return km
    return None

# -------------------------
# Load artifacts / snapshot / pipeline
# -------------------------
snapshot_df = get_snapshot_df()  # may be None
pipeline = get_pipeline()        # may be None
label_classes = get_label_classes(pipeline)
vectorizer, sim_matrix = get_vectorizer_and_sim()
kmeans_model = get_kmeans()

# If snapshot exists, ensure combined_notes exists
if snapshot_df is not None and "combined_notes" not in snapshot_df.columns:
    # try to construct combined_notes from top/mid/base if present
    cols = [c for c in ["top notes","mid notes","base notes"] if c in snapshot_df.columns]
    if cols:
        snapshot_df["combined_notes"] = snapshot_df[cols].fillna("").agg(", ".join, axis=1)
        snapshot_df["combined_notes"] = snapshot_df["combined_notes"].apply(clean_text_notes)

# -------------------------
# Main UI Tabs
# -------------------------
tabs = st.tabs(["Overview", "Visualizations", "Recommend", "Predict (Classifier)", "Artifacts"])
tab_overview, tab_viz, tab_reco, tab_pred, tab_art = tabs

with tab_overview:
    st.header("Overview")
    st.markdown("""
    This app **deploys** pre-built artifacts (no training inside the app).
    Upload artifacts in the sidebar or place them into an artifact folder and point to it.
    """)
    if snapshot_df is None:
        st.warning("No preprocessed snapshot loaded. Upload 'preprocessed_snapshot.csv' to enable recommender & visualizations.")
    else:
        st.subheader("Snapshot preview")
        st.dataframe(snapshot_df.head(10))
        st.write("Rows:", snapshot_df.shape[0], "Columns:", snapshot_df.shape[1])

with tab_viz:
    st.header("Visualizations (from snapshot)")
    if snapshot_df is None:
        st.info("Upload snapshot to see visualizations.")
    else:
        df = snapshot_df.copy()
        # quick metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Unique Brands", int(df['brand'].nunique()) if 'brand' in df.columns else 0)
        c3.metric("Avg price per ml", float(df['price_per_ml'].median()) if 'price_per_ml' in df.columns else 0)

        # histograms
        fig, ax = plt.subplots(1,3, figsize=(16,4))
        if 'price' in df.columns:
            sns.histplot(df['price'].dropna(), bins=40, kde=True, ax=ax[0])
            ax[0].set_title("Price distribution")
        if 'size' in df.columns:
            sns.histplot(df['size'].dropna(), bins=30, kde=False, ax=ax[1])
            ax[1].set_title("Size distribution")
        if 'price_per_ml' in df.columns:
            sns.histplot(df['price_per_ml'].dropna(), bins=40, kde=True, ax=ax[2])
            ax[2].set_title("Price per ML")
        st.pyplot(fig)
        plt.close(fig)

        # corr heatmap numeric
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        # top notes barplot
        if 'combined_notes' in df.columns:
            tokens = []
            for s in df['combined_notes'].fillna(""):
                tokens.extend([t.strip() for t in s.split(",") if t.strip()!=""])
            top = Counter(tokens).most_common(20)
            labels, counts = zip(*top) if top else ([], [])
            if top:
                fig, ax = plt.subplots(figsize=(8,6))
                sns.barplot(x=list(counts), y=list(labels), ax=ax)
                ax.set_title("Top 20 Notes")
                st.pyplot(fig)
                plt.close(fig)

        # wordcloud
        if HAS_WORDCLOUD and 'combined_notes' in df.columns:
            st.subheader("WordCloud (combined notes)")
            text_all = " ".join(df['combined_notes'].fillna("").tolist())
            wc = generate_wordcloud_image(text_all)
            if wc:
                fig = plt.figure(figsize=(10,5))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(fig)
                plt.close(fig)
        elif not HAS_WORDCLOUD:
            st.info("Install `wordcloud` to view wordcloud visualization.")

with tab_reco:
    st.header("Content-based Recommender (based on notes)")
    st.markdown("Use the pre-built vectorizer & (optional) sim matrix. If vectorizer not provided, the app will fit TF-IDF on the snapshot (not training the classifier).")

    if snapshot_df is None:
        st.info("Upload snapshot to use recommender.")
    else:
        df = snapshot_df.copy()
        query = st.text_input("Query notes (e.g. 'vanilla amber musk')", value="vanilla amber")
        k = st.slider("Top-K", 1, 20, 5)
        # filters
        st.markdown("Optional filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            situation_filter = st.selectbox("situation (empty = no filter)", [""] + sorted([str(x) for x in df['situation'].dropna().unique()]) if 'situation' in df.columns else [""], index=0)
        with col2:
            brand_filter = st.text_input("Brand (exact match, empty = no filter)", value="")
        with col3:
            price_max = st.number_input("Price max (0 = no limit)", value=0.0)
        # build or use vectorizer
        vect = vectorizer
        simm = sim_matrix
        if vect is None:
            st.warning("No prebuilt vectorizer artifact found. App will fit TF-IDF on snapshot (fast for small datasets).")
            vect = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words="english")
            Xvec = vect.fit_transform(df['combined_notes'].fillna(""))
            simm = None
        # prepare candidate pool
        candidates = df.copy()
        if situation_filter:
            candidates = candidates[candidates['situation'].astype(str).str.lower()==situation_filter.lower()]
        if brand_filter:
            candidates = candidates[candidates['brand'].astype(str).str.lower()==brand_filter.lower()]
        if price_max > 0:
            candidates = candidates[candidates['price'] <= price_max]
        if candidates.shape[0] == 0:
            st.info("No candidates after filters.")
        else:
            # compute recommendations
            qvec = vect.transform([query])
            cand_texts = candidates['combined_notes'].fillna("").tolist()
            cand_vecs = vect.transform(cand_texts)
            scores = cosine_similarity(qvec, cand_vecs).flatten()
            res = []
            for i, sc in enumerate(scores):
                r = candidates.iloc[i]
                res.append({"perfume": r.get("perfume"), "brand": r.get("brand"), "price": r.get("price"), "score": float(sc), "notes": r.get("combined_notes")})
            res_sorted = sorted(res, key=lambda x: x["score"], reverse=True)[:k]
            st.write(f"Top {len(res_sorted)} recommendations:")
            for r in res_sorted:
                st.write(f"- **{r['perfume']}** (brand: {r['brand']}) — price: {r['price']} — score: {r['score']:.4f}")
                st.caption(r['notes'])

with tab_pred:
    st.header("Prediction (deployed classifier inference)")
    st.markdown("This uses the pre-loaded `classifier_pipeline.joblib`. If not available, prediction UI is disabled.")
    if pipeline is None:
        st.info("No pipeline loaded. Upload `classifier_pipeline.joblib` in sidebar or place into artifact dir.")
    else:
        st.subheader("Single sample predict")
        st.markdown("You can input `combined_notes` + optional categorical/numeric fields. The pipeline will do preprocessing and predict label.")
        # Build input form
        with st.form("predict_form"):
            notes_in = st.text_area("combined_notes (comma-separated notes)", value="vanilla, amber, musk")
            brand_in = st.text_input("brand (optional)", value="")
            concentrate_in = st.text_input("concentrate (optional)", value="")
            gender_in = st.selectbox("gender (optional)", ["", "Unisex", "Female", "Male"])
            price_in = st.number_input("price (optional)", value=0.0, step=1000.0)
            size_in = st.number_input("size (optional)", value=0, step=1)
            submitted = st.form_submit_button("Predict")
        if submitted:
            # Build feature dict matching pipeline expected columns: combined_notes + categorical + numeric
            feat = {}
            feat['combined_notes'] = notes_in
            # add categorical columns only if pipeline expects them: we will attempt to detect names in pipeline's ColumnTransformer
            X_input = pd.DataFrame([feat])
            # Add other possible columns if pipeline built earlier with them
            # Try to detect from pipeline.named_steps['preproc']
            try:
                preproc = pipeline.named_steps.get("preproc", None)
                # find column names expected by the preprocessor
                # if ColumnTransformer, attributes .transformers_ available
                if preproc is not None and hasattr(preproc, "transformers_"):
                    # collect names of columns in transformers that are 'remainder' drop
                    cols_needed = []
                    for name, trans, col_spec in preproc.transformers_:
                        # col_spec can be string, list, slice, or callable; handle list
                        if isinstance(col_spec, (list, tuple)):
                            cols_needed.extend(col_spec)
                        elif isinstance(col_spec, str):
                            cols_needed.append(col_spec)
                    # ensure X_input contains these columns; set defaults
                    for c in cols_needed:
                        if c not in X_input.columns:
                            if c == "brand":
                                X_input[c] = brand_in
                            elif c == "concentrate":
                                X_input[c] = concentrate_in
                            elif c == "gender":
                                X_input[c] = gender_in
                            elif c == "price_log":
                                # pipeline expects price_log numeric; compute from price_in
                                X_input[c] = np.log1p(price_in)
                            elif c == "price_per_ml_clip":
                                X_input[c] = (price_in / max(1, size_in)) if size_in > 0 else 0
                            elif c == "size":
                                X_input[c] = size_in
                            else:
                                X_input[c] = ""
                else:
                    # fallback: add common columns
                    X_input["brand"] = brand_in
                    X_input["concentrate"] = concentrate_in
                    X_input["gender"] = gender_in
                    X_input["price_log"] = np.log1p(price_in)
                    X_input["price_per_ml_clip"] = (price_in / max(1, size_in)) if size_in>0 else 0
                    X_input["size"] = size_in
            except Exception as e:
                st.warning(f"Failed to detect pipeline feature schema: {e}")
                # fill naive defaults
                X_input["brand"] = brand_in
                X_input["concentrate"] = concentrate_in
                X_input["gender"] = gender_in
                X_input["price_log"] = np.log1p(price_in)
                X_input["price_per_ml_clip"] = (price_in / max(1, size_in)) if size_in>0 else 0
                X_input["size"] = size_in

            # Run prediction
            try:
                pred = pipeline.predict(X_input)
                if label_classes:
                    # map
                    if isinstance(label_classes, list):
                        pred_label = label_classes[int(pred[0])]
                    else:
                        pred_label = str(pred[0])
                else:
                    pred_label = str(pred[0])
                st.success(f"Predicted: {pred_label} (encoded: {pred[0]})")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab_art:
    st.header("Loaded Artifacts")
    st.write("Artifact dir:", str(artifact_dir) if artifact_dir else "None")
    st.write("Pipeline loaded:", "Yes" if pipeline is not None else "No")
    st.write("Label classes:", label_classes if label_classes is not None else "None")
    st.write("Snapshot loaded:", "Yes" if snapshot_df is not None else "No")
    st.write("Vectorizer:", "Yes" if vectorizer is not None else "No")
    st.write("Sim matrix:", "Yes" if sim_matrix is not None else "No")
    st.write("KMeans model:", "Yes" if kmeans_model is not None else "No")

st.markdown("---")
st.caption("This app performs inference / recommendation only. For training pipeline, please run training script offline and place artifacts into artifact folder.")