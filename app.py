import io
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

TRAIN_PATH    = Path(__file__).parent / "DS_smartPhone.csv"
CATEGORY_COLS = [
    "Payment_Method", "Website_Activity", "Bought_Digital_Media_18Mo",
    "Gender", "Bought_Electronics_12Mo", "Bought_Digital",
    "Browsed_Electronics_12Mo", "Marital_Status",
]
NUMERIC_COLS  = ["Age"]
TARGET        = "Smartphone_Adoption"
PRIORITY_SEGS = ["Innovator", "Early Adopter"]
ORDER         = ["Innovator", "Early Adopter", "Early Majority", "Late Majority"]
PALETTE       = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"]
THRESHOLD     = {"F1 (weighted)": 0.75, "Precision": 0.75, "Recall": 0.50, "ROC-AUC": 0.75}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Segmentasi Smartphone", layout="wide")
st.title("Segmentasi Pelanggan Smartphone")
st.caption("CRISP-DM — Fase Deployment | Model: Gradient Boosting")

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file, sep=";")
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def load_and_train():
    df = pd.read_csv(TRAIN_PATH, delimiter=";")
    X  = df.drop([TARGET, "User_ID"], axis=1)
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORY_COLS),
    ])
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=1),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
         "max_depth": [3, 5], "subsample": [0.8, 1.0]},
        cv=cv, scoring="f1_weighted", n_jobs=-1,
    )
    gs.fit(X_train_t, y_train)

    model = gs.best_estimator_
    yp    = model.predict(X_test_t)
    pr    = model.predict_proba(X_test_t)
    metrics = {
        "F1 (weighted)": f1_score(y_test, yp, average="weighted"),
        "Precision":     precision_score(y_test, yp, average="weighted"),
        "Recall":        recall_score(y_test, yp, average="weighted"),
        "ROC-AUC":       roc_auc_score(y_test, pr, multi_class="ovr", average="weighted"),
    }
    return preprocessor, model, metrics, gs.best_params_


def score(preprocessor, model, df_new):
    X_new   = preprocessor.transform(df_new.drop("User_ID", axis=1))
    proba   = model.predict_proba(X_new)
    result  = df_new.copy()
    result["Predicted_Segment"] = model.predict(X_new)
    for i, c in enumerate(model.classes_):
        result[f"Conf_{c}"] = proba[:, i].round(3)
    result["Max_Confidence"] = proba.max(axis=1).round(3)
    return result


def dist_chart(dist_pred, dist_pct):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars = axes[0].bar(ORDER, [dist_pred.get(s, 0) for s in ORDER],
                       color=PALETTE, edgecolor="white")
    for bar, seg in zip(bars, ORDER):
        n, p = dist_pred.get(seg, 0), dist_pct.get(seg, 0)
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{n}\n({p:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("Jumlah per Segmen", fontweight="bold")
    axes[0].set_ylabel("Jumlah Pelanggan")
    axes[0].set_ylim(0, max(dist_pred.values(), default=1) * 1.25)
    axes[0].tick_params(axis="x", rotation=10)
    axes[1].pie([dist_pred.get(s, 0) for s in ORDER], labels=ORDER, colors=PALETTE,
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Proporsi Segmen", fontweight="bold")
    plt.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Data Baru")
    new_file = st.file_uploader("Data Calon Pembeli (CSV, sep=;)", type="csv")
    st.caption("Data training sudah tersimpan di sistem.")

# ── Guard ─────────────────────────────────────────────────────────────────────
if not TRAIN_PATH.exists():
    st.error(f"File training tidak ditemukan: `{TRAIN_PATH.name}`. "
             "Letakkan file tersebut di folder yang sama dengan app.py.")
    st.stop()

# ── Load model (cached) ───────────────────────────────────────────────────────
with st.spinner("Memuat model... (hanya sekali, hasil di-cache)"):
    preprocessor, model, metrics, best_params = load_and_train()

# ── Tabs — struktur deployment sesuai PPT ────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Plan Deployment",
    "Implement Model",
    "Documentation & Reports",
    "Review",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Plan Deployment
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Plan Deployment")
    st.markdown(
        "Menentukan cara penggunaan model dan langkah teknis implementasinya."
    )
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Tujuan Deployment**")
        st.markdown(
            "- Memprediksi segmen adopsi calon pembeli smartphone baru\n"
            "- Mengidentifikasi **Innovator** dan **Early Adopter** sebagai target prioritas marketing\n"
            "- Menghasilkan daftar pelanggan yang siap dikirim ke tim marketing"
        )
        st.markdown("**Alur Sistem**")
        st.code(
            "Upload data calon pembeli (.csv)\n"
            "       ↓\n"
            "Preprocessing (StandardScaler + OneHotEncoder)\n"
            "       ↓\n"
            "Model Gradient Boosting → prediksi segmen\n"
            "       ↓\n"
            "Output: tabel segmen + confidence score + download",
            language=None,
        )

    with col_b:
        st.markdown("**Spesifikasi Model**")
        spec = {
            "Algoritma":       "Gradient Boosting Classifier",
            "Optimasi":        "GridSearchCV (5-Fold Stratified CV)",
            "Scoring":         "F1-Score (weighted)",
            "Data Training":   "5.000 baris historis",
            "Fitur Input":     f"{len(CATEGORY_COLS)} kategorikal + {len(NUMERIC_COLS)} numerik",
            "Kelas Output":    ", ".join(ORDER),
        }
        st.table(pd.DataFrame(spec.items(), columns=["Parameter", "Nilai"]))

        st.markdown("**Best Hyperparameters**")
        st.json(best_params)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Implement Model
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Implement Model")
    st.markdown(
        "Integrasikan model ke sistem produksi — upload data calon pembeli "
        "untuk mendapatkan prediksi segmen secara langsung."
    )
    st.divider()

    if new_file is None:
        st.info("Upload **Data Calon Pembeli** di sidebar untuk memulai scoring.")
        st.stop()

    df_new    = load_csv(new_file)
    df_result = score(preprocessor, model, df_new)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total calon pembeli", f"{len(df_new):,}")
    priority_n = len(df_result[df_result["Predicted_Segment"].isin(PRIORITY_SEGS)])
    m2.metric("Target prioritas (Innovator + EA)", f"{priority_n:,}")
    m3.metric("Confidence rata-rata", f"{df_result['Max_Confidence'].mean():.2f}")

    st.divider()
    st.markdown("**Hasil Scoring — Semua Calon Pembeli**")
    show_cols = ["User_ID", "Age", "Gender", "Payment_Method",
                 "Website_Activity", "Predicted_Segment", "Max_Confidence"]
    st.dataframe(df_result[show_cols].reset_index(drop=True), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Documentation & Reports
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Documentation & Reports")
    st.markdown(
        "Dokumentasi model dan laporan interpretasi hasil scoring untuk stakeholder."
    )
    st.divider()

    if new_file is None:
        st.info("Upload **Data Calon Pembeli** di sidebar terlebih dahulu.")
        st.stop()

    # Distribusi segmen
    dist_pred = df_result["Predicted_Segment"].value_counts().reindex(ORDER, fill_value=0).to_dict()
    dist_pct  = {k: v / len(df_result) * 100 for k, v in dist_pred.items()}

    st.markdown("**Distribusi Prediksi Segmen**")
    fig = dist_chart(dist_pred, dist_pct)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Profil segmen
    st.markdown("**Profil per Segmen**")
    profile_rows = []
    for seg in ORDER:
        sub = df_result[df_result["Predicted_Segment"] == seg]
        if len(sub) == 0:
            continue
        profile_rows.append({
            "Segmen":          seg,
            "Jumlah":          len(sub),
            "Proporsi (%)":    f"{len(sub)/len(df_result)*100:.1f}%",
            "Rata-rata Usia":  f"{sub['Age'].mean():.0f} thn",
            "Gender Dominan":  sub["Gender"].value_counts().index[0],
            "Payment Dominan": sub["Payment_Method"].value_counts().index[0],
            "Web Activity":    sub["Website_Activity"].value_counts().index[0],
            "Conf. Rata-rata": f"{sub['Max_Confidence'].mean():.2f}",
        })
    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Target prioritas
    priority = df_result[df_result["Predicted_Segment"].isin(PRIORITY_SEGS)].sort_values(
        ["Predicted_Segment", "Max_Confidence"], ascending=[True, False]
    )
    st.markdown(f"**Daftar Target Prioritas — Innovator + Early Adopter ({len(priority)} orang)**")
    st.dataframe(priority[show_cols].reset_index(drop=True), use_container_width=True)

    st.divider()

    # Download
    st.markdown("**Download Laporan**")
    d1, d2 = st.columns(2)
    with d1:
        buf = io.StringIO()
        df_result.to_csv(buf, index=False)
        st.download_button("Download semua pelanggan (scoring lengkap)",
                           data=buf.getvalue(),
                           file_name="hasil_scoring_pelanggan.csv",
                           mime="text/csv")
    with d2:
        buf2 = io.StringIO()
        priority[show_cols].reset_index(drop=True).to_csv(buf2, index=False)
        st.download_button("Download target prioritas (Innovator + Early Adopter)",
                           data=buf2.getvalue(),
                           file_name="target_prioritas_marketing.csv",
                           mime="text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Review
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Review")
    st.markdown(
        "Evaluasi hasil terhadap business objectives dan rekomendasi tindak lanjut."
    )
    st.divider()

    # Metrik vs threshold
    st.markdown("**Performa Model vs Business Objectives**")
    summary_rows = []
    for k, v in metrics.items():
        thr = THRESHOLD[k]
        summary_rows.append({
            "Metrik":    k,
            "Nilai":     round(v, 4),
            "Threshold": thr,
            "Status":    "✓ PASS" if v >= thr else "✗ FAIL",
        })
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(
        df_summary.style.map(
            lambda x: "color: green; font-weight: bold" if x == "✓ PASS"
            else ("color: red; font-weight: bold" if x == "✗ FAIL" else ""),
            subset=["Status"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Rekomendasi marketing
    st.markdown("**Rekomendasi Aksi Tim Marketing**")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**Innovator + Early Adopter**")
        st.caption("Aksi segera")
        st.markdown(
            "- Akses early bird / pre-order eksklusif\n"
            "- Undangan beta tester atau program ambassador\n"
            "- Channel: email personal, WhatsApp\n"
            "- *\"Jadilah yang pertama memiliki produk ini\"*"
        )
    with r2:
        st.markdown("**Early Majority**")
        st.caption("2 minggu setelah launch")
        st.markdown(
            "- Kirim bukti sosial dari Innovator/Early Adopter\n"
            "- Channel: email blast, retargeting iklan\n"
            "- *\"Sudah [N] pelanggan membuktikannya\"*"
        )
    with r3:
        st.markdown("**Late Majority**")
        st.caption("Setelah stok masih ada")
        st.markdown(
            "- Trigger: diskon, bundle, cicilan 0%\n"
            "- Jangan habiskan anggaran sebelum high-priority terlayani"
        )

    st.divider()

    # Kesimpulan
    if new_file is not None:
        total_new    = len(df_new)
        priority_pct = priority_n / total_new * 100
        st.info(
            f"Total high-priority target (Innovator + Early Adopter): **{priority_n} orang** "
            f"({priority_pct:.1f}% dari {total_new} calon pembeli).  \n"
            "Fokuskan anggaran marketing di segmen ini untuk menciptakan efek wave adoption.  \n"
            "File `target_prioritas_marketing.csv` siap dikirimkan ke tim marketing."
        )
    else:
        st.info("Upload data calon pembeli untuk melihat kesimpulan hasil scoring.")
