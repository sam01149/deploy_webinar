import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

CATEGORY_COLS = [
    "Payment_Method", "Website_Activity", "Bought_Digital_Media_18Mo",
    "Gender", "Bought_Electronics_12Mo", "Bought_Digital",
    "Browsed_Electronics_12Mo", "Marital_Status",
]
NUMERIC_COLS = ["Age"]
TARGET = "Smartphone_Adoption"
PRIORITY_SEGS = ["Innovator", "Early Adopter"]
ORDER = ["Innovator", "Early Adopter", "Early Majority", "Late Majority"]
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"]

st.set_page_config(page_title="Segmentasi Smartphone", layout="wide", page_icon="📱")
st.title("📱 Segmentasi Pelanggan Smartphone")
st.caption("Model: Gradient Boosting — CRISP-DM Deployment")


@st.cache_data
def load_csv(file, sep=";"):
    df = pd.read_csv(file, sep=sep)
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def train_pipeline(_df_hash, _df):
    X = _df.drop([TARGET, "User_ID"], axis=1)
    y = _df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORY_COLS),
    ])
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    }
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=1),
        param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1,
    )
    gs.fit(X_train_t, y_train)

    final_model = gs.best_estimator_
    yp = final_model.predict(X_test_t)
    pr = final_model.predict_proba(X_test_t)

    metrics = {
        "F1 (weighted)": f1_score(y_test, yp, average="weighted"),
        "Precision":     precision_score(y_test, yp, average="weighted"),
        "Recall":        recall_score(y_test, yp, average="weighted"),
        "ROC-AUC":       roc_auc_score(y_test, pr, multi_class="ovr", average="weighted"),
    }

    return preprocessor, final_model, metrics, gs.best_params_


def score_new_data(preprocessor, model, df_new):
    X_new = preprocessor.transform(df_new.drop("User_ID", axis=1))
    classes = model.classes_

    df_result = df_new.copy()
    df_result["Predicted_Segment"] = model.predict(X_new)
    proba = model.predict_proba(X_new)
    for i, c in enumerate(classes):
        df_result[f"Conf_{c}"] = proba[:, i].round(3)
    df_result["Max_Confidence"] = proba.max(axis=1).round(3)
    return df_result


def plot_segment_distribution(dist_pred, dist_pct):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bars = axes[0].bar(ORDER, [dist_pred.get(s, 0) for s in ORDER],
                       color=PALETTE, edgecolor="white")
    for bar, seg in zip(bars, ORDER):
        n = dist_pred.get(seg, 0)
        p = dist_pct.get(seg, 0)
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{n}\n({p:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("Jumlah per Segmen Prediksi", fontweight="bold")
    axes[0].set_ylabel("Jumlah Pelanggan")
    axes[0].set_ylim(0, max(dist_pred.values()) * 1.25)
    axes[0].tick_params(axis="x", rotation=10)

    axes[1].pie(
        [dist_pred.get(s, 0) for s in ORDER],
        labels=ORDER, colors=PALETTE, autopct="%1.1f%%",
        startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Proporsi Segmen", fontweight="bold")
    plt.tight_layout()
    return fig


# ── Sidebar: upload ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Data")
    train_file = st.file_uploader("Data Training (CSV, sep=;)", type="csv", key="train")
    new_file   = st.file_uploader("Data Calon Pembeli (CSV, sep=;)", type="csv", key="new")

if train_file is None or new_file is None:
    st.info("Upload **Data Training** dan **Data Calon Pembeli** di sidebar untuk memulai.")
    st.stop()

# ── Load data ────────────────────────────────────────────────────────────────
df_train = load_csv(train_file)
df_new   = load_csv(new_file)

col1, col2 = st.columns(2)
col1.metric("Baris data training", f"{len(df_train):,}")
col2.metric("Calon pembeli baru", f"{len(df_new):,}")

st.divider()

# ── Train model ──────────────────────────────────────────────────────────────
with st.spinner("Melatih model Gradient Boosting (GridSearchCV)... ini membutuhkan ~1 menit."):
    df_hash = hash(df_train.to_json())
    preprocessor, final_model, metrics, best_params = train_pipeline(df_hash, df_train)

# ── Metrics ──────────────────────────────────────────────────────────────────
st.subheader("Performa Model (Test Set)")
mcols = st.columns(4)
threshold = {"F1 (weighted)": 0.75, "Precision": 0.75, "Recall": 0.50, "ROC-AUC": 0.75}
for col, (k, v) in zip(mcols, metrics.items()):
    status = "✓" if v >= threshold[k] else "✗"
    col.metric(f"{status} {k}", f"{v:.4f}", delta=f"threshold {threshold[k]}")

with st.expander("Best hyperparameters"):
    st.json(best_params)

st.divider()

# ── Scoring ──────────────────────────────────────────────────────────────────
st.subheader("Hasil Scoring — Calon Pembeli Baru")
df_result = score_new_data(preprocessor, final_model, df_new)

dist_pred = df_result["Predicted_Segment"].value_counts().reindex(ORDER, fill_value=0)
dist_pct  = (dist_pred / len(df_result) * 100).to_dict()
dist_pred = dist_pred.to_dict()

fig = plot_segment_distribution(dist_pred, dist_pct)
st.pyplot(fig)
plt.close(fig)

st.divider()

# ── Priority targets ─────────────────────────────────────────────────────────
priority = df_result[df_result["Predicted_Segment"].isin(PRIORITY_SEGS)].sort_values(
    ["Predicted_Segment", "Max_Confidence"], ascending=[True, False]
)

st.subheader(f"Target Prioritas Tinggi — Innovator + Early Adopter ({len(priority)} orang)")
show_cols = [
    "User_ID", "Age", "Gender", "Payment_Method", "Website_Activity",
    "Predicted_Segment", "Max_Confidence",
]
st.dataframe(priority[show_cols].reset_index(drop=True), use_container_width=True)

st.divider()

# ── Segment profiles ─────────────────────────────────────────────────────────
st.subheader("Profil per Segmen")
profile_rows = []
for seg in ORDER:
    sub = df_result[df_result["Predicted_Segment"] == seg]
    if len(sub) == 0:
        continue
    profile_rows.append({
        "Segmen":           seg,
        "Jumlah":           len(sub),
        "Proporsi (%)":     f"{len(sub)/len(df_result)*100:.1f}%",
        "Rata-rata Usia":   f"{sub['Age'].mean():.0f} thn",
        "Gender Dominan":   sub["Gender"].value_counts().index[0],
        "Payment Dominan":  sub["Payment_Method"].value_counts().index[0],
        "Web Activity":     sub["Website_Activity"].value_counts().index[0],
        "Conf. Rata-rata":  f"{sub['Max_Confidence'].mean():.2f}",
    })
st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

st.divider()

# ── Download ─────────────────────────────────────────────────────────────────
st.subheader("Download Hasil")
dcol1, dcol2 = st.columns(2)

with dcol1:
    buf = io.StringIO()
    df_result.to_csv(buf, index=False)
    st.download_button(
        "⬇ Semua pelanggan (scoring lengkap)",
        data=buf.getvalue(),
        file_name="hasil_scoring_pelanggan.csv",
        mime="text/csv",
    )

with dcol2:
    buf2 = io.StringIO()
    priority[show_cols].reset_index(drop=True).to_csv(buf2, index=False)
    st.download_button(
        "⬇ Target prioritas (Innovator + Early Adopter)",
        data=buf2.getvalue(),
        file_name="target_prioritas_marketing.csv",
        mime="text/csv",
    )

st.divider()

# ── Fase 6.7 — Rekomendasi Tim Marketing ─────────────────────────────────────
st.subheader("6.7 Rekomendasi untuk Tim Marketing")

rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    st.markdown("**1. Innovator + Early Adopter — Aksi Segera**")
    st.markdown(
        "- Berikan akses early bird / pre-order eksklusif\n"
        "- Kirim undangan beta tester atau program ambassador\n"
        "- Channel: email personal, WhatsApp langsung\n"
        "- Pesan: *\"Jadilah yang pertama memiliki produk ini\"*"
    )

with rec_col2:
    st.markdown("**2. Early Majority — Aksi dalam 2 minggu setelah launch**")
    st.markdown(
        "- Tunggu review dari Innovator/Early Adopter, lalu kirim bukti sosial\n"
        "- Channel: email blast, retargeting iklan\n"
        "- Pesan: *\"Sudah [N] pelanggan kami membuktikannya\"*"
    )

with rec_col3:
    st.markdown("**3. Late Majority — Aksi setelah stok masih ada**")
    st.markdown(
        "- Trigger: diskon, bundle, cicilan 0%\n"
        "- Jangan habiskan anggaran sebelum high-priority terlayani"
    )

st.divider()

# ── Fase 7 — Ringkasan Akhir ─────────────────────────────────────────────────
st.subheader("7. Ringkasan Akhir")

THRESHOLD = {"F1 (weighted)": 0.75, "Precision": 0.75, "Recall": 0.50, "ROC-AUC": 0.75}

summary_rows = []
for k, v in metrics.items():
    thr = THRESHOLD[k]
    summary_rows.append({
        "Metrik": k,
        "Nilai": round(v, 4),
        "Threshold": thr,
        "Status": "✓ PASS" if v >= thr else "✗ FAIL",
    })
df_summary = pd.DataFrame(summary_rows)

st.markdown("**Model Final: Gradient Boosting (Tuned)**")
st.dataframe(
    df_summary.style.applymap(
        lambda x: "color: green; font-weight: bold" if x == "✓ PASS"
        else ("color: red; font-weight: bold" if x == "✗ FAIL" else ""),
        subset=["Status"],
    ),
    use_container_width=True,
    hide_index=True,
)

priority_count = len(priority)
total_new = len(df_new)
st.info(
    f"Total high-priority target (Innovator + Early Adopter): **{priority_count} orang** "
    f"({priority_count/total_new*100:.1f}% dari total {total_new} calon pembeli).\n\n"
    "Fokuskan anggaran marketing di segmen ini untuk menciptakan efek wave adoption. "
    "File `target_prioritas_marketing.csv` siap dikirimkan ke tim marketing."
)
