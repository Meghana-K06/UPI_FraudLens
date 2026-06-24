"""
FraudSight Enhanced Dashboard — Phase 2 & 3
─────────────────────────────────────────────
Features:
  Phase 1: Random Forest ML detection, analytics charts
  Phase 2: Device fingerprint risk, velocity detection,
           first-time beneficiary alerts, unusual hour detection
  Phase 3: SHAP-inspired Explainable AI (XAI) module
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import time

# Import our engine
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from fraud_engine import (
    VelocityTracker, BeneficiaryTracker,
    generate_device_fingerprint, assess_device_risk,
    is_unusual_hour, compute_risk_contributions,
    compute_composite_risk, get_transaction_hour,
    RISK_COLORS, reset_device_registry
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FraudSight",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Core palette ── */
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface2: #1c2128;
  --border: #30363d;
  --text: #e6edf3;
  --text-muted: #8b949e;
  --accent: #58a6ff;
  --accent2: #3fb950;
  --danger: #FF2D55;
  --warning: #FFB800;
  --safe: #34C759;
}

/* Global reset */
.stApp { background: var(--bg); color: var(--text); }
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* Hide default header */
header[data-testid="stHeader"] { display: none; }

/* ── Custom page header ── */
.fs-header {
  display: flex; align-items: center; gap: 1rem;
  padding: 1.5rem 2rem;
  background: linear-gradient(135deg, #0d1b2a 0%, #1a1f3a 50%, #0d1117 100%);
  border-bottom: 1px solid var(--border);
  margin: -2rem -2.5rem 2rem;
}
.fs-logo { font-size: 2.4rem; }
.fs-title { font-size: 1.8rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
.fs-subtitle { font-size: 0.85rem; color: var(--text-muted); margin-top: 2px; }
.fs-badge {
  margin-left: auto;
  background: rgba(88, 166, 255, 0.15);
  border: 1px solid rgba(88, 166, 255, 0.4);
  color: var(--accent);
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  position: relative; overflow: hidden;
}
.metric-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: var(--accent-color, var(--accent));
}
.metric-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }
.metric-value { font-size: 2rem; font-weight: 700; color: var(--text); line-height: 1; }
.metric-sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.4rem; }

/* ── Risk badge ── */
.risk-badge {
  display: inline-block;
  padding: 0.3rem 1rem;
  border-radius: 20px;
  font-weight: 700;
  font-size: 0.9rem;
  letter-spacing: 0.05em;
}

/* ── XAI Contribution bar ── */
.xai-row {
  display: flex; align-items: center; gap: 0.75rem;
  padding: 0.6rem 0;
  border-bottom: 1px solid rgba(48,54,61,0.5);
}
.xai-icon { font-size: 1.2rem; width: 1.8rem; text-align: center; }
.xai-label { flex: 0 0 180px; font-size: 0.85rem; color: var(--text); }
.xai-bar-wrap { flex: 1; background: rgba(48,54,61,0.5); border-radius: 4px; height: 8px; }
.xai-bar { height: 8px; border-radius: 4px; }
.xai-value { flex: 0 0 50px; font-size: 0.85rem; font-weight: 600; text-align: right; }
.xai-desc { font-size: 0.78rem; color: var(--text-muted); flex: 0 0 280px; }

/* ── Alert banners ── */
.alert-box {
  border-radius: 10px; padding: 0.8rem 1.2rem;
  margin: 0.4rem 0; font-size: 0.88rem;
  display: flex; align-items: flex-start; gap: 0.6rem;
  border: 1px solid;
}
.alert-critical { background: rgba(255,45,85,0.12); border-color: rgba(255,45,85,0.4); color: #ff6b8a; }
.alert-high { background: rgba(255,107,53,0.12); border-color: rgba(255,107,53,0.4); color: #ff9066; }
.alert-medium { background: rgba(255,184,0,0.12); border-color: rgba(255,184,0,0.4); color: #ffd060; }
.alert-safe { background: rgba(52,199,89,0.12); border-color: rgba(52,199,89,0.4); color: #5dd879; }

/* ── Section headers ── */
.section-header {
  font-size: 1.05rem; font-weight: 600; color: var(--text);
  padding: 0.5rem 0; margin-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 0.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div {
  background: var(--surface2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

/* Matplotlib dark style override */
.stPlotlyChart, .stImage { border-radius: 10px; overflow: hidden; }

/* Expander */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-muted) !important;
  padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(88,166,255,0.15) !important;
  border-color: rgba(88,166,255,0.5) !important;
  color: var(--accent) !important;
}

/* Data table */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }

/* Score dial */
.score-dial {
  text-align: center; padding: 1.5rem;
  background: var(--surface); border-radius: 16px;
  border: 1px solid var(--border);
}
.score-number { font-size: 3.5rem; font-weight: 800; line-height: 1; }
.score-label { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="fs-header">
  <div class="fs-logo">🛡️</div>
  <div>
    <div class="fs-title">FraudSight</div>
    <div class="fs-subtitle">UPI Transaction Intelligence Platform</div>
  </div>
  <div class="fs-badge">Live Analysis</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(path: str = "fraud_data_sample.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hour"] = df["step"] % 24
    df["is_night"] = df["hour"].apply(lambda h: 1 if (h < 5 or h >= 22) else 0)
    df["balance_drop"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_unchanged"] = ((df["newbalanceDest"] - df["oldbalanceDest"]) == 0).astype(int)
    df["account_drained"] = (df["newbalanceOrig"] == 0).astype(int)
    return df


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    le = LabelEncoder()
    df2 = df.copy()
    df2["type_enc"] = le.fit_transform(df2["type"])

    features = ["type_enc", "amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest", "balance_drop",
                "dest_unchanged", "account_drained", "is_night"]
    X = df2[features]
    y = df2["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=120, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    return model, le, features, auc, X_test, y_test


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["📊 Analytics Overview", "🔍 Transaction Inspector", "🧠 XAI Explorer", "📈 Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🎚️ Risk Thresholds")
    velocity_threshold = st.slider("Velocity alert (tx/2h)", 2, 10, 5)
    fraud_threshold = st.slider("ML fraud threshold", 0.3, 0.9, 0.5, 0.05)
    amount_filter = st.slider("Min amount (₹)", 0, 500_000, 0, 10_000)

    st.markdown("---")
    st.markdown("### 🔎 Filter")
    tx_types_all = ["ALL", "TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    selected_type = st.selectbox("Transaction Type", tx_types_all)

    st.markdown("---")
    st.caption("FraudSight v2.0 · Phase 2+3 Active")


# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────

with st.spinner("🔄 Loading transaction data..."):
    df = load_data()

with st.spinner("🤖 Training Random Forest model..."):
    model, le, features, auc, X_test, y_test = train_model(df)

# Apply filters
df_view = df.copy()
if selected_type != "ALL":
    df_view = df_view[df_view["type"] == selected_type]
if amount_filter > 0:
    df_view = df_view[df_view["amount"] >= amount_filter]


# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#161b22",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#8b949e",
    "text.color": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
})


# ═══════════════════════════════════════════════
# PAGE 1: ANALYTICS OVERVIEW
# ═══════════════════════════════════════════════

if nav == "📊 Analytics Overview":
    # ── KPI cards ──
    total_tx = len(df_view)
    total_fraud = df_view["isFraud"].sum()
    fraud_rate = (total_fraud / total_tx * 100) if total_tx else 0
    total_amount = df_view["amount"].sum()
    fraud_amount = df_view[df_view["isFraud"] == 1]["amount"].sum()
    night_fraud = df_view[(df_view["isFraud"] == 1) & (df_view["is_night"] == 1)].shape[0]
    night_pct = (night_fraud / total_fraud * 100) if total_fraud else 0

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card" style="--accent-color:#58a6ff">
        <div class="metric-label">Total Transactions</div>
        <div class="metric-value">{total_tx:,}</div>
        <div class="metric-sub">₹{total_amount/1e7:.1f} Cr total value</div>
      </div>
      <div class="metric-card" style="--accent-color:#FF2D55">
        <div class="metric-label">Fraud Detected</div>
        <div class="metric-value">{total_fraud:,}</div>
        <div class="metric-sub">₹{fraud_amount/1e7:.2f} Cr at risk</div>
      </div>
      <div class="metric-card" style="--accent-color:#FFB800">
        <div class="metric-label">Fraud Rate</div>
        <div class="metric-value">{fraud_rate:.2f}%</div>
        <div class="metric-sub">of all transactions</div>
      </div>
      <div class="metric-card" style="--accent-color:#9c7cd4">
        <div class="metric-label">Night-time Fraud</div>
        <div class="metric-value">{night_pct:.0f}%</div>
        <div class="metric-sub">of fraud is 10PM–5AM</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row 1 ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Fraud by Transaction Type</div>', unsafe_allow_html=True)
        fraud_by_type = df_view.groupby("type")["isFraud"].agg(["sum", "mean"]).reset_index()
        fraud_by_type.columns = ["type", "count", "rate"]
        fraud_by_type["rate"] *= 100

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#FF2D55" if t in ("TRANSFER", "CASH_OUT") else "#58a6ff" for t in fraud_by_type["type"]]
        bars = ax.bar(fraud_by_type["type"], fraud_by_type["count"], color=colors, width=0.55)
        ax.set_ylabel("Fraud Count")
        ax.set_title("Fraudulent Transactions by Type", color="#e6edf3", pad=10)
        for bar, rate in zip(bars, fraud_by_type["rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=8, color="#8b949e")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">⏰ Fraud by Hour of Day</div>', unsafe_allow_html=True)
        hourly = df_view.groupby("hour")["isFraud"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.fill_between(hourly["hour"], hourly["isFraud"],
                        alpha=0.3, color="#FF2D55")
        ax.plot(hourly["hour"], hourly["isFraud"],
                color="#FF2D55", linewidth=2)
        ax.axvspan(0, 5, alpha=0.12, color="#9c7cd4", label="High-risk hours")
        ax.axvspan(22, 24, alpha=0.12, color="#9c7cd4")
        ax.set_xlabel("Hour (0-23)")
        ax.set_ylabel("Fraud Count")
        ax.set_title("When Does Fraud Happen?", color="#e6edf3", pad=10)
        ax.legend(fontsize=8, facecolor="#1c2128", edgecolor="#30363d", labelcolor="#8b949e")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Charts row 2 ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">💸 Amount Distribution</div>', unsafe_allow_html=True)
        fraud_amounts = df_view[df_view["isFraud"] == 1]["amount"]
        legit_amounts = df_view[df_view["isFraud"] == 0]["amount"].sample(min(5000, len(df_view[df_view["isFraud"] == 0])))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(np.log1p(legit_amounts), bins=60, alpha=0.5, color="#58a6ff", label="Legitimate")
        ax.hist(np.log1p(fraud_amounts), bins=60, alpha=0.7, color="#FF2D55", label="Fraudulent")
        ax.set_xlabel("log(Amount + 1)")
        ax.set_ylabel("Frequency")
        ax.set_title("Amount Distribution: Fraud vs Legitimate", color="#e6edf3", pad=10)
        ax.legend(fontsize=8, facecolor="#1c2128", edgecolor="#30363d", labelcolor="#8b949e")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown('<div class="section-header">📈 Fraud Over Time</div>', unsafe_allow_html=True)
        fraud_time = df_view.groupby("step")["isFraud"].sum().rolling(window=12).mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(fraud_time["step"], fraud_time["isFraud"], color="#FFB800", linewidth=1.5)
        ax.fill_between(fraud_time["step"], fraud_time["isFraud"], alpha=0.2, color="#FFB800")
        ax.set_xlabel("Time (Steps)")
        ax.set_ylabel("Fraud Count (12-step MA)")
        ax.set_title("Fraud Trend Over Time", color="#e6edf3", pad=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Heatmap ──
    st.markdown('<div class="section-header">🔥 Fraud Heatmap: Type × Hour</div>', unsafe_allow_html=True)
    pivot = df_view.groupby(["type", "hour"])["isFraud"].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 3))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.5,
                linecolor="#0d1117", cbar_kws={"shrink": 0.8})
    ax.set_title("Fraud Concentration by Type & Hour of Day", color="#e6edf3", pad=10)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Transaction Type")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════
# PAGE 2: TRANSACTION INSPECTOR
# ═══════════════════════════════════════════════

elif nav == "🔍 Transaction Inspector":
    st.markdown("### 🔍 Real-time Transaction Risk Analyzer")
    st.markdown("Enter transaction details below to get an instant multi-layer risk assessment.")

    # Session state for velocity and beneficiary trackers
    if "velocity_tracker" not in st.session_state:
        st.session_state.velocity_tracker = VelocityTracker(window_hours=2, threshold=velocity_threshold)
        st.session_state.beneficiary_tracker = BeneficiaryTracker()
        st.session_state.tx_history = []
        reset_device_registry()

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">📝 Transaction Details</div>', unsafe_allow_html=True)

        with st.form("tx_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                sender = st.text_input("Sender Account", value="C1231006815")
                tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
                step = st.number_input("Time Step (hour)", min_value=0, max_value=744, value=14)
            with col_b:
                recipient = st.text_input("Recipient Account", value="C553264065")
                amount = st.number_input("Amount (₹)", min_value=1.0, value=45000.0, step=1000.0)

            col_c, col_d = st.columns(2)
            with col_c:
                old_bal_orig = st.number_input("Sender Old Balance (₹)", value=50000.0, step=1000.0)
                new_bal_orig = st.number_input("Sender New Balance (₹)", value=5000.0, step=1000.0)
            with col_d:
                old_bal_dest = st.number_input("Recipient Old Balance (₹)", value=0.0, step=1000.0)
                new_bal_dest = st.number_input("Recipient New Balance (₹)", value=0.0, step=1000.0)

            submit = st.form_submit_button("🔎 Analyze Transaction", use_container_width=True)

        # Session reset
        if st.button("🔄 Reset Session (clear velocity/device history)"):
            st.session_state.velocity_tracker = VelocityTracker(window_hours=2, threshold=velocity_threshold)
            st.session_state.beneficiary_tracker = BeneficiaryTracker()
            st.session_state.tx_history = []
            reset_device_registry()
            st.success("Session reset.")

    with col_result:
        if submit:
            tx = {
                "step": step, "type": tx_type, "amount": amount,
                "nameOrig": sender, "nameDest": recipient,
                "oldbalanceOrg": old_bal_orig, "newbalanceOrig": new_bal_orig,
                "oldbalanceDest": old_bal_dest, "newbalanceDest": new_bal_dest,
            }

            # ── ML Prediction ──
            try:
                type_enc = le.transform([tx_type])[0]
            except Exception:
                type_enc = 0

            balance_drop = old_bal_orig - new_bal_orig
            dest_unchanged = int((new_bal_dest - old_bal_dest) == 0)
            account_drained = int(new_bal_orig == 0)
            is_night = int(step % 24 < 5 or step % 24 >= 22)

            feature_vec = pd.DataFrame([[
                type_enc, amount, old_bal_orig, new_bal_orig,
                old_bal_dest, new_bal_dest, balance_drop,
                dest_unchanged, account_drained, is_night
            ]], columns=features)

            ml_prob = float(model.predict_proba(feature_vec)[0][1])

            # ── Phase 2 Signals ──
            vel_flagged, vel_reason, vel_count = st.session_state.velocity_tracker.record_and_check(
                sender, step, amount
            )
            is_new_bene, bene_reason = st.session_state.beneficiary_tracker.check_and_register(
                sender, recipient, amount
            )
            device_fp = generate_device_fingerprint(sender, amount, step)
            dev_risk, dev_reason = assess_device_risk(sender, device_fp)
            unusual_time, hour_reason = is_unusual_hour(step)

            # ── Composite Score ──
            composite, risk_level = compute_composite_risk(
                ml_prob, dev_risk, vel_flagged, is_new_bene,
                unusual_time, amount, tx_type
            )

            color = RISK_COLORS[risk_level]

            # ── Risk Score Display ──
            st.markdown(f"""
            <div class="score-dial">
              <div class="score-number" style="color:{color}">{composite:.0%}</div>
              <div style="margin-top:0.5rem">
                <span class="risk-badge" style="background:{color}22;color:{color};border:1px solid {color}66">
                  ⚠ {risk_level} RISK
                </span>
              </div>
              <div class="score-label">Composite Fraud Probability</div>
              <div style="color:#8b949e;font-size:0.8rem;margin-top:0.5rem">
                ML: {ml_prob:.0%} &nbsp;|&nbsp; Device: {dev_risk:.0%} &nbsp;|&nbsp; Hour: {step%24:02d}:00
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Alert Flags ──
            st.markdown("#### 🚨 Active Alerts")
            alerts_html = ""
            if risk_level in ("CRITICAL", "HIGH"):
                alerts_html += f'<div class="alert-box alert-critical">🔴 <strong>ML Model:</strong> {ml_prob:.0%} fraud probability — model flags this transaction</div>'
            if vel_flagged:
                alerts_html += f'<div class="alert-box alert-critical">⚡ <strong>Velocity:</strong> {vel_reason}</div>'
            if is_new_bene:
                alerts_html += f'<div class="alert-box alert-high">👤 <strong>Beneficiary:</strong> {bene_reason}</div>'
            if dev_risk > 0.5:
                alerts_html += f'<div class="alert-box alert-high">📱 <strong>Device:</strong> {dev_reason}</div>'
            if unusual_time:
                alerts_html += f'<div class="alert-box alert-medium">🌙 <strong>Timing:</strong> {hour_reason}</div>'
            if not alerts_html:
                alerts_html = '<div class="alert-box alert-safe">✅ <strong>No critical flags</strong> — transaction appears normal</div>'

            st.markdown(alerts_html, unsafe_allow_html=True)

            # Store in history
            tx["risk_score"] = composite
            tx["risk_level"] = risk_level
            tx["ml_prob"] = ml_prob
            st.session_state.tx_history.append(tx)

        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;color:#8b949e">
              <div style="font-size:3rem">🔎</div>
              <div style="font-size:1.1rem;margin-top:1rem">Fill in transaction details and click Analyze</div>
              <div style="font-size:0.85rem;margin-top:0.5rem">All Phase 2 & 3 signals will be evaluated</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Transaction History Table ──
    if st.session_state.get("tx_history"):
        st.markdown("---")
        st.markdown('<div class="section-header">📋 Analysis Session History</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.tx_history)[
            ["nameOrig", "nameDest", "type", "amount", "step", "ml_prob", "risk_score", "risk_level"]
        ]
        hist_df.columns = ["Sender", "Recipient", "Type", "Amount (₹)", "Step", "ML Prob", "Risk Score", "Level"]
        hist_df["Amount (₹)"] = hist_df["Amount (₹)"].apply(lambda x: f"₹{x:,.0f}")
        hist_df["ML Prob"] = hist_df["ML Prob"].apply(lambda x: f"{x:.0%}")
        hist_df["Risk Score"] = hist_df["Risk Score"].apply(lambda x: f"{x:.0%}")
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# PAGE 3: XAI EXPLORER
# ═══════════════════════════════════════════════

elif nav == "🧠 XAI Explorer":
    st.markdown("### 🧠 Explainable AI — Why Was This Flagged?")
    st.markdown("Select a transaction from the dataset to understand exactly what drove the fraud score.")

    # Pick a sample from high-risk fraud cases
    fraud_sample = df[df["isFraud"] == 1].sample(min(200, df["isFraud"].sum()), random_state=42)
    legit_sample = df[df["isFraud"] == 0].sample(50, random_state=42)
    xai_pool = pd.concat([fraud_sample, legit_sample]).reset_index(drop=True)

    col_pick, col_explain = st.columns([1, 1.6], gap="large")

    with col_pick:
        st.markdown('<div class="section-header">🔢 Pick a Transaction</div>', unsafe_allow_html=True)

        idx = st.slider("Transaction index", 0, len(xai_pool) - 1, 0)
        row = xai_pool.iloc[idx]

        # Show transaction card
        is_fraud_label = "🚨 FRAUD" if row["isFraud"] else "✅ LEGIT"
        fraud_color = "#FF2D55" if row["isFraud"] else "#34C759"

        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.2rem">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem">
            <span style="font-weight:700;font-size:1rem">Transaction #{idx}</span>
            <span class="risk-badge" style="background:{fraud_color}22;color:{fraud_color};border:1px solid {fraud_color}55">
              {is_fraud_label}
            </span>
          </div>
          <table style="width:100%;font-size:0.85rem;border-collapse:collapse">
            <tr><td style="color:#8b949e;padding:3px 0">Type</td><td style="text-align:right;font-weight:600">{row['type']}</td></tr>
            <tr><td style="color:#8b949e;padding:3px 0">Amount</td><td style="text-align:right;font-weight:600">₹{row['amount']:,.0f}</td></tr>
            <tr><td style="color:#8b949e;padding:3px 0">Sender</td><td style="text-align:right">{row['nameOrig']}</td></tr>
            <tr><td style="color:#8b949e;padding:3px 0">Recipient</td><td style="text-align:right">{row['nameDest']}</td></tr>
            <tr><td style="color:#8b949e;padding:3px 0">Time (Step)</td><td style="text-align:right">{row['step']} (Hour {int(row['step'])%24:02d}:00)</td></tr>
            <tr><td style="color:#8b949e;padding:3px 0">Sender Balance</td><td style="text-align:right">₹{row['oldbalanceOrg']:,.0f} → ₹{row['newbalanceOrig']:,.0f}</td></tr>
            <tr><td style="color:#8b949e;padding:3px 0">Dest Balance</td><td style="text-align:right">₹{row['oldbalanceDest']:,.0f} → ₹{row['newbalanceDest']:,.0f}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        # ML prediction for this transaction
        try:
            type_enc = le.transform([row["type"]])[0]
        except Exception:
            type_enc = 0

        balance_drop = row["oldbalanceOrg"] - row["newbalanceOrig"]
        dest_unchanged = int((row["newbalanceDest"] - row["oldbalanceDest"]) == 0)
        account_drained = int(row["newbalanceOrig"] == 0)
        is_night = int(int(row["step"]) % 24 < 5 or int(row["step"]) % 24 >= 22)

        fv = pd.DataFrame([[
            type_enc, row["amount"], row["oldbalanceOrg"], row["newbalanceOrig"],
            row["oldbalanceDest"], row["newbalanceDest"], balance_drop,
            dest_unchanged, account_drained, is_night
        ]], columns=features)

        ml_prob = float(model.predict_proba(fv)[0][1])

        # Simulate contextual signals for XAI
        vel_count = np.random.randint(1, 8) if row["isFraud"] else np.random.randint(1, 3)
        is_new_bene = bool(row["isFraud"]) or bool(np.random.random() < 0.15)
        dev_risk = (0.6 + np.random.random() * 0.3) if row["isFraud"] else (np.random.random() * 0.2)
        unusual_time = bool(is_night)

        composite, risk_level = compute_composite_risk(
            ml_prob, dev_risk, vel_count >= velocity_threshold,
            is_new_bene, unusual_time, row["amount"], row["type"]
        )

        color = RISK_COLORS[risk_level]
        st.markdown(f"""
        <div style="text-align:center;margin-top:1rem;padding:1rem;
                    background:#161b22;border:1px solid {color}44;border-radius:10px">
          <div style="font-size:2.2rem;font-weight:800;color:{color}">{composite:.0%}</div>
          <div style="font-size:0.8rem;color:#8b949e">Composite Risk · ML: {ml_prob:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_explain:
        st.markdown('<div class="section-header">🔬 Risk Factor Breakdown</div>', unsafe_allow_html=True)

        contributions = compute_risk_contributions(
            row.to_dict(), ml_prob, vel_count, is_new_bene, dev_risk, unusual_time
        )

        total_pos = sum(c["contribution"] for c in contributions if c["contribution"] > 0)
        total_neg = sum(abs(c["contribution"]) for c in contributions if c["contribution"] < 0)

        xai_html = ""
        for c in contributions:
            val = c["contribution"]
            pct = abs(val) / max(total_pos, total_neg, 1) * 100
            bar_color = "#FF2D55" if val > 0 else "#34C759"
            sign = "+" if val > 0 else ""
            xai_html += f"""
            <div class="xai-row">
              <div class="xai-icon">{c['icon']}</div>
              <div class="xai-label">{c['factor']}</div>
              <div class="xai-bar-wrap">
                <div class="xai-bar" style="width:{pct}%;background:{bar_color}"></div>
              </div>
              <div class="xai-value" style="color:{bar_color}">{sign}{val}</div>
              <div class="xai-desc">{c['description']}</div>
            </div>
            """

        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.2rem">
          <div style="display:flex;gap:1.5rem;margin-bottom:1rem;font-size:0.82rem">
            <span>🔴 Risk factors (push toward fraud)</span>
            <span>🟢 Safety factors (push toward legit)</span>
          </div>
          {xai_html}
        </div>
        """, unsafe_allow_html=True)

        # Contribution chart
        st.markdown('<div class="section-header" style="margin-top:1.5rem">📊 Visual Contribution Chart</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, max(3, len(contributions) * 0.55)))
        labels = [c["factor"] for c in contributions]
        vals = [c["contribution"] for c in contributions]
        colors_bar = ["#FF2D55" if v > 0 else "#34C759" for v in vals]

        y_pos = range(len(labels))
        bars = ax.barh(y_pos, vals, color=colors_bar, height=0.6, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="#30363d", linewidth=1)
        ax.set_xlabel("Risk Contribution (arbitrary units)")
        ax.set_title(f"SHAP-style Explanation — Risk Level: {risk_level}", color="#e6edf3", pad=10)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        for bar, val in zip(bars, vals):
            x = bar.get_width()
            ax.text(x + (1 if val >= 0 else -1), bar.get_y() + bar.get_height() / 2,
                    f"{'+' if val>0 else ''}{val}", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8, color="#8b949e")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(f"""
        <div style="background:#1c2128;border:1px solid #30363d;border-radius:10px;
                    padding:1rem;margin-top:0.75rem;font-size:0.85rem;color:#8b949e">
          <strong style="color:#e6edf3">🧠 Model Interpretation:</strong><br>
          This transaction received a <strong style="color:{RISK_COLORS[risk_level]}">{risk_level}</strong>
          risk rating of <strong style="color:{RISK_COLORS[risk_level]}">{composite:.0%}</strong>.
          The dominant risk signals were
          <strong style="color:#e6edf3">{contributions[0]['factor']}</strong>
          ({'+' if contributions[0]['contribution'] > 0 else ''}{contributions[0]['contribution']} pts) and
          <strong style="color:#e6edf3">{contributions[1]['factor'] if len(contributions) > 1 else 'N/A'}</strong>
          ({'+' if len(contributions) > 1 and contributions[1]['contribution'] > 0 else ''}{contributions[1]['contribution'] if len(contributions) > 1 else 0} pts).
          {'Account drainage and TRANSFER type are the strongest fraud predictors in this model.' if risk_level in ('CRITICAL','HIGH') else 'No single dominant fraud signal was found — composite risk is low.'}
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ═══════════════════════════════════════════════

elif nav == "📈 Model Performance":
    st.markdown("### 📈 Random Forest — Model Metrics")

    # ── Score cards ──
    y_pred = (model.predict_proba(X_test)[:, 1] >= fraud_threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)

    precision = report.get("1", {}).get("precision", 0)
    recall = report.get("1", {}).get("recall", 0)
    f1 = report.get("1", {}).get("f1-score", 0)

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card" style="--accent-color:#58a6ff">
        <div class="metric-label">AUC-ROC Score</div>
        <div class="metric-value">{auc:.4f}</div>
        <div class="metric-sub">Higher = better discrimination</div>
      </div>
      <div class="metric-card" style="--accent-color:#3fb950">
        <div class="metric-label">Precision (Fraud)</div>
        <div class="metric-value">{precision:.2%}</div>
        <div class="metric-sub">Of flagged, how many are real fraud</div>
      </div>
      <div class="metric-card" style="--accent-color:#FFB800">
        <div class="metric-label">Recall (Fraud)</div>
        <div class="metric-value">{recall:.2%}</div>
        <div class="metric-sub">Of all fraud, how many caught</div>
      </div>
      <div class="metric-card" style="--accent-color:#FF2D55">
        <div class="metric-label">F1 Score</div>
        <div class="metric-value">{f1:.4f}</div>
        <div class="metric-sub">Harmonic mean P/R</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🌳 Feature Importance</div>', unsafe_allow_html=True)
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors_fi = ["#FF2D55" if i > feat_imp.quantile(0.6) else "#58a6ff" for i in feat_imp.values]
        ax.barh(feat_imp.index, feat_imp.values, color=colors_fi, height=0.65, edgecolor="none")
        ax.set_xlabel("Feature Importance")
        ax.set_title("Random Forest Feature Importances", color="#e6edf3", pad=10)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">📊 Score Distribution</div>', unsafe_allow_html=True)
        y_scores = model.predict_proba(X_test)[:, 1]
        fraud_scores = y_scores[y_test == 1]
        legit_scores = y_scores[y_test == 0]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(legit_scores, bins=60, alpha=0.6, color="#58a6ff", label="Legitimate", density=True)
        ax.hist(fraud_scores, bins=60, alpha=0.7, color="#FF2D55", label="Fraudulent", density=True)
        ax.axvline(fraud_threshold, color="#FFB800", linestyle="--", linewidth=1.5,
                   label=f"Threshold ({fraud_threshold})")
        ax.set_xlabel("Fraud Probability Score")
        ax.set_ylabel("Density")
        ax.set_title("ML Score Distributions", color="#e6edf3", pad=10)
        ax.legend(fontsize=8, facecolor="#1c2128", edgecolor="#30363d", labelcolor="#8b949e")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Model config
    st.markdown('<div class="section-header">⚙️ Model Configuration</div>', unsafe_allow_html=True)
    cfg = {
        "Algorithm": "Random Forest Classifier",
        "Estimators": 120,
        "Max Depth": 12,
        "Class Weight": "Balanced (handles class imbalance)",
        "Training Split": "80% train / 20% test",
        "Phase 2 Signals": "Velocity · Beneficiary · Device · Hour",
        "XAI Method": "Additive Risk Decomposition (SHAP-inspired)",
    }
    cfg_df = pd.DataFrame(list(cfg.items()), columns=["Parameter", "Value"])
    st.dataframe(cfg_df, use_container_width=True, hide_index=True)
