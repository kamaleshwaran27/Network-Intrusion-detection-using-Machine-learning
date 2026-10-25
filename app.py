import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

RANDOM_STATE = 42
NORMAL_LABELS = {"normal", "benign", "0", "false", "no", "non-attack", "non attack", "normal traffic", "benign traffic"}

st.set_page_config(page_title="NetShield | Intrusion Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #07111f 0%, #0b1728 55%, #101d32 100%); color: #e7eef8;}
    [data-testid="stSidebar"] {background: #0b1422;}
    [data-testid="stMetric"] {background: #101f33; border: 1px solid #203854; padding: 16px 18px; border-radius: 14px;}
    [data-testid="stMetricLabel"] {color: #91a9c4;}
    [data-testid="stMetricValue"] {color: #f1f7ff;}
    h1, h2, h3 {color: #f2f7ff;}
    .hero {padding: 24px 28px; border: 1px solid #203854; border-radius: 18px; background: linear-gradient(120deg, #10243a, #112d40); margin-bottom: 18px;}
    .hero small {color: #76e4c4; font-weight: 700; letter-spacing: 1.6px;}
    .hero p {color: #a9bdd4; margin-bottom: 0;}
    .status-pill {display:inline-block; padding:5px 11px; border-radius:20px; background:#183a3c; color:#7df0ce; font-size:12px; font-weight:700;}
    div[data-testid="stDataFrame"] {border: 1px solid #203854; border-radius: 10px;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <small>NETWORK SECURITY • MACHINE LEARNING</small>
  <h1>🛡️ NetShield Intrusion Detection</h1>
  <p>Analyze network-flow records, compare machine-learning models, and flag potentially suspicious traffic.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛡️ NetShield")
    st.caption("Detection workspace")
    uploaded_file = st.file_uploader("Upload labelled traffic CSV", type=["csv"])
    st.markdown("---")
    st.markdown("### Training configuration")
    test_size = st.slider("Test split", 10, 40, 20, 5) / 100
    selected_models = st.multiselect(
        "Algorithms",
        ["Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "Dummy baseline"],
        default=["Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees"]
    )
    st.markdown("---")
    st.caption("Prototype • For academic evaluation")

if uploaded_file is None:
    st.subheader("Welcome to your security dashboard")
    st.write("Upload your `network_intrusion_synthetic_500.csv` file in the sidebar to populate the dashboard.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset status", "Waiting")
    c2.metric("Models", "5 available")
    c3.metric("Prediction mode", "CSV")
    st.info("Expected input: a labelled CSV with network traffic features and a target column such as `Label`.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("CSV must contain records, feature columns, and a label column.")
    st.stop()

target_candidates = ["label", "labels", "class", "target", "attack", "category", "is_attack", "outcome"]
normalized_columns = {str(c).strip().lower(): c for c in df.columns}
detected_target = next((normalized_columns[c] for c in target_candidates if c in normalized_columns), None)
target_options = list(df.columns)
default_index = target_options.index(detected_target) if detected_target in target_options else len(target_options) - 1

st.markdown("### 📡 Traffic overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Traffic records", f"{len(df):,}")
k2.metric("Total features", f"{len(df.columns)-1:,}")
k3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
k4.metric("Duplicate records", f"{int(df.duplicated().sum()):,}")

with st.expander("Inspect uploaded data"):
    st.dataframe(df.head(25), use_container_width=True)
    st.dataframe(pd.DataFrame({
        "Data type": df.dtypes.astype(str),
        "Missing values": df.isna().sum(),
        "Unique values": df.nunique(dropna=True)
    }), use_container_width=True)

target_column = st.selectbox("Target / label column", target_options, index=default_index)
work = df.copy()
work[target_column] = work[target_column].apply(lambda x: x.strip() if isinstance(x, str) else x)
work = work.dropna(subset=[target_column]).copy()
raw_labels = work[target_column].astype(str).str.strip()
y = (~raw_labels.str.lower().isin(NORMAL_LABELS)).astype(int)
y.name = "target_binary"
X = work.drop(columns=[target_column]).copy()
all_missing = X.columns[X.isna().all()].tolist()
constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
index_like = [c for c in X.columns if str(c).strip().lower() in {"unnamed: 0", "index", "flow_id", "flowid"}]
drop_cols = sorted(set(all_missing + constant + index_like))
X = X.drop(columns=drop_cols, errors="ignore").replace([np.inf, -np.inf], np.nan)

if X.shape[1] == 0:
    st.error("No usable feature columns found.")
    st.stop()
if y.nunique() < 2:
    st.error("Target labels map to only one class. Check the selected target and NORMAL_LABELS.")
    st.stop()
if y.value_counts().min() < 2:
    st.error("Each class needs at least two records for stratified train/test split.")
    st.stop()

counts = y.map({0: "Normal", 1: "Suspicious"}).value_counts().reindex(["Normal", "Suspicious"], fill_value=0)
left, right = st.columns([1, 1.7])
with left:
    st.markdown("#### Traffic classification")
    st.dataframe(counts.rename_axis("Class").to_frame("Records"), use_container_width=True)
with right:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Records")
    ax.set_title("Normal vs suspicious traffic")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.markdown("### 🧠 Model training and evaluation")
st.write("Choose the algorithms in the sidebar, then train them on a stratified train/test split.")
train_clicked = st.button("🚀 Train models & build report", type="primary", use_container_width=False)

if train_clicked:
    if not selected_models:
        st.warning("Select at least one algorithm in the sidebar.")
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
            )
            numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
            categorical_features = [c for c in X_train.columns if c not in numeric_features]
            transformers = []
            if numeric_features:
                transformers.append(("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), numeric_features))
            if categorical_features:
                transformers.append(("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
                ]), categorical_features))
            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            catalog = {
                "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
                "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE, min_samples_leaf=2),
                "Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
                "Extra Trees": ExtraTreesClassifier(n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
                "Dummy baseline": DummyClassifier(strategy="most_frequent")
            }
            rows, fitted, predictions = [], {}, {}
            progress = st.progress(0, text="Preparing model training...")
            for i, name in enumerate(selected_models):
                pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", catalog[name])])
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
                fitted[name] = pipe
                predictions[name] = pred
                rows.append({
                    "Algorithm": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, zero_division=0),
                    "Recall": recall_score(y_test, pred, zero_division=0),
                    "F1-score": f1_score(y_test, pred, zero_division=0)
                })
                progress.progress((i + 1) / len(selected_models), text=f"Trained {name}")
            results = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
            best_name = results.iloc[0]["Algorithm"]
            best_model = fitted[best_name]
            best_pred = predictions[best_name]
            buffer = io.BytesIO()
            joblib.dump(best_model, buffer)
            buffer.seek(0)
            st.session_state["results"] = results
            st.session_state["best_name"] = best_name
            st.session_state["model_bytes"] = buffer.getvalue()
            st.session_state["feature_columns"] = list(X.columns)
            st.session_state["test_data"] = {
                "cm": confusion_matrix(y_test, best_pred, labels=[0, 1]),
                "report": classification_report(y_test, best_pred, labels=[0, 1],
                    target_names=["Normal", "Suspicious"], zero_division=0),
                "y_test": y_test.to_numpy(),
                "pred": np.asarray(best_pred)
            }
            st.success("Training complete. Dashboard metrics updated below.")
        except Exception as exc:
            st.error(f"Training failed: {type(exc).__name__}: {exc}")

if "results" in st.session_state:
    results = st.session_state["results"]
    best = results.iloc[0]
    td = st.session_state["test_data"]
    st.markdown("---")
    st.markdown("### 📊 Model performance")
    a, b, c, d = st.columns(4)
    a.metric("Best model", st.session_state["best_name"])
    b.metric("Accuracy", f"{best['Accuracy']:.1%}")
    c.metric("Recall", f"{best['Recall']:.1%}")
    d.metric("F1-score", f"{best['F1-score']:.1%}")

    tab1, tab2, tab3 = st.tabs(["Performance comparison", "Confusion matrix", "Detailed metrics"])
    with tab1:
        shown = results.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
            shown[col] = shown[col].map(lambda v: f"{v:.2%}")
        st.dataframe(shown, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        results.set_index("Algorithm")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar", ax=ax, ylim=(0, 1))
        ax.set_ylabel("Score")
        ax.set_title("Model comparison on held-out test data")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.download_button("⬇️ Download metrics CSV", results.to_csv(index=False).encode("utf-8"),
            "model_comparison.csv", "text/csv")
    with tab2:
        cm = td["cm"]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm)
        ax.set_xticks([0, 1], labels=["Normal", "Suspicious"])
        ax.set_yticks([0, 1], labels=["Normal", "Suspicious"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix — best F1-score")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("True negatives", int(tn))
        x2.metric("False positives", int(fp))
        x3.metric("False negatives", int(fn))
        x4.metric("True positives", int(tp))
    with tab3:
        st.code(td["report"])
        st.download_button("⬇️ Download trained model", st.session_state["model_bytes"],
            "network_intrusion_model.joblib", "application/octet-stream")

    st.markdown("---")
    st.markdown("### 🔎 Predict traffic from another CSV")
    st.caption("Upload feature records using the same feature columns as training. The label column is optional and will be ignored.")
    prediction_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="prediction_upload")
    if prediction_file is not None:
        try:
            new_data = pd.read_csv(prediction_file).drop(columns=[target_column], errors="ignore")
            expected = st.session_state["feature_columns"]
            for col in expected:
                if col not in new_data.columns:
                    new_data[col] = np.nan
            new_data = new_data[expected].replace([np.inf, -np.inf], np.nan)
            model = joblib.load(io.BytesIO(st.session_state["model_bytes"]))
            pred = model.predict(new_data)
            output = new_data.copy()
            output["Prediction"] = np.where(pred == 0, "Normal", "Suspicious")
            st.dataframe(output, use_container_width=True)
            st.download_button("⬇️ Download prediction results", output.to_csv(index=False).encode("utf-8"),
                "intrusion_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

st.markdown("---")
st.caption("NetShield academic prototype • Synthetic data results are not a guarantee of real-world detection performance.")
