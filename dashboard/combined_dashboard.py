import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from adaptive_model.adaptive_training import retrain_with_new_user, retrain_all
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = 'data'

# -------------------- PAGE CONFIG --------------------
st.set_page_config(layout="wide")
st.title("AI-Powered Insider Threat Detection: Department-aware Dashboard")

# -------------------- CONFIG: department-specific guidance thresholds --------------------
DEPT_FEATURE_THRESHOLDS = {
    "Customer Care": {
        "mean_login_hour": (6, 22),
        "files_per_day": (0, 200),
        "usb_per_day": (0, 1),
        "emails_per_day": (0, 300)
    },
    "IT Support": {
        "mean_login_hour": (6, 22),
        "files_per_day": (0, 300),
        "usb_per_day": (0, 6),
        "emails_per_day": (0, 100)
    },
    "Finance": {
        "mean_login_hour": (8, 18),
        "files_per_day": (0, 100),
        "usb_per_day": (0, 2),
        "emails_per_day": (0, 80)
    },
    "Unknown": {
        "mean_login_hour": (0, 23),
        "files_per_day": (0, 200),
        "usb_per_day": (0, 5),
        "emails_per_day": (0, 200)
    }
}

# -------------------- LOAD DATA --------------------
def load_all_data():
    features_path = os.path.join(DATA_DIR, "merged_features.csv")
    scores_path = os.path.join(DATA_DIR, "anomaly_scores.csv")

    features = pd.read_csv(features_path) if os.path.exists(features_path) else pd.DataFrame()
    scores = pd.read_csv(scores_path) if os.path.exists(scores_path) else pd.DataFrame()

    if "department" not in features.columns:
        features["department"] = "Unknown"

    # Merge department info into scores if missing
    if not scores.empty and "department" not in scores.columns and "user" in scores.columns:
        scores = scores.merge(features[["user", "department"]], on="user", how="left")

    return features, scores


features, scores = load_all_data()
if "department" not in features.columns:
    features["department"] = "Unknown"

if not scores.empty and not features.empty:
    df = pd.merge(features, scores, on="user", how="left")
else:
    df = features.copy()

# -------------------- HELPER: risk level --------------------
def compute_risk_levels(df_display):
    if "aggregated_score" not in df_display.columns:
        if "isolation_forest" in df_display.columns:
            df_display["aggregated_score"] = df_display["isolation_forest"]
        else:
            df_display["aggregated_score"] = 0.0

    df_display["aggregated_score"] = df_display["aggregated_score"].fillna(0.0)
    df_display["Risk Level"] = "🟢 Low Risk"

    for dept in df_display["department"].fillna("Unknown").unique():
        mask = df_display["department"].fillna("Unknown") == dept
        sub = df_display.loc[mask, "aggregated_score"]
        if sub.empty:
            continue
        q90, q70 = sub.quantile(0.90), sub.quantile(0.70)
        df_display.loc[mask & (df_display["aggregated_score"] >= q90), "Risk Level"] = "🔴 High Risk"
        df_display.loc[mask & (df_display["aggregated_score"] < q90) & (df_display["aggregated_score"] >= q70), "Risk Level"] = "🟡 Medium Risk"
        df_display.loc[mask & (df_display["aggregated_score"] < q70), "Risk Level"] = "🟢 Low Risk"

    return df_display


# -------------------- UI TABS --------------------
anomaly_tab, user_tab, adaptive_tab, how_tab = st.tabs([
    "Anomaly Table", "User Detail", "Adaptive Learning", "How It Works"
])

# -------------------- ANOMALY TABLE --------------------
with anomaly_tab:
    st.header("Department-wise Anomaly Scores")

    departments = ["All"] + sorted(DEPT_FEATURE_THRESHOLDS.keys())
    selected_department = st.selectbox("Filter by Department", departments, index=0)

    df_display = df.copy()
    if selected_department != "All":
        df_display = df_display[df_display["department"] == selected_department].copy()

    df_display = compute_risk_levels(df_display)

    feature_cols = ["mean_login_hour", "files_per_day", "usb_per_day", "emails_per_day"]
    show_cols = ["user", "department", "aggregated_score", "Risk Level"] + [c for c in feature_cols if c in df_display.columns]
    st.dataframe(df_display[show_cols].fillna(""), height=480)

    st.subheader("Top 5 Users (by aggregated score)")
    top5 = df_display.sort_values("aggregated_score", ascending=False).head(5)
    if not top5.empty:
        st.bar_chart(top5.set_index("user")["aggregated_score"])

# -------------------- USER DETAIL --------------------
with user_tab:
    st.header("User Detail")
    if df.empty:
        st.info("No users found. Add a new user from the Adaptive Learning tab.")
    else:
        selected_user = st.selectbox("Select a User", df["user"].unique())
        row = df[df["user"] == selected_user].iloc[0]
        st.markdown(f"**Department:** {row.get('department','Unknown')}")
        st.markdown(f"**Aggregated Score:** {row.get('aggregated_score', 0.0):.3f}")
        st.markdown(f"**Risk Level:** {compute_risk_levels(pd.DataFrame([row]))['Risk Level'].iloc[0]}")

        st.write("### Feature Overview")
        feat_cols = ["mean_login_hour", "files_per_day", "usb_per_day", "emails_per_day"]
        st.json({c: row.get(c, None) for c in feat_cols})

# -------------------- ADAPTIVE LEARNING --------------------
with adaptive_tab:
    st.header("Adaptive Learning — Manage Department Users")

    # -------- Add User --------
    st.subheader("➕ Add New User")
    with st.form("add_user_form"):
        new_user = {}
        new_user["user"] = st.text_input("User ID (unique)", value="")
        dept = st.selectbox("Department", sorted(DEPT_FEATURE_THRESHOLDS.keys()), index=0)
        new_user["department"] = dept

        thr = DEPT_FEATURE_THRESHOLDS.get(dept, DEPT_FEATURE_THRESHOLDS["Unknown"])
        st.markdown(f"**Guidance for {dept}:**")
        st.markdown(f"- Typical login hour: {thr['mean_login_hour']}")
        st.markdown(f"- Files/day: {thr['files_per_day']}")
        st.markdown(f"- USB/day: {thr['usb_per_day']}")
        st.markdown(f"- Emails/day: {thr['emails_per_day']}")

        new_user["mean_login_hour"] = float(st.number_input("Average Login Hour (0–23)", 0.0, 23.0, float(np.mean(thr["mean_login_hour"]))))
        new_user["files_per_day"] = int(st.number_input("Files Per Day", 0, 10000, int(np.mean(thr["files_per_day"]))))
        new_user["usb_per_day"] = float(st.number_input("USB Devices Per Day", 0.0, 100.0, float(np.mean(thr["usb_per_day"]))))
        new_user["emails_per_day"] = int(st.number_input("Emails Sent Per Day", 0, 10000, int(np.mean(thr["emails_per_day"]))))

        submitted = st.form_submit_button("Add User & Retrain")

    if submitted:
        if not new_user["user"].strip():
            st.error("Please enter a valid user ID.")
        else:
            st.info(f"Adding user to {new_user['department']} and retraining model...")
            result = retrain_with_new_user(new_user)
            st.success(f"✅ {result['user']} added. Aggregated Score = {result['aggregated_score']:.3f}")
            st.write(f"Department thresholds: q90 = {result['q90']:.3f}, q70 = {result['q70']:.3f}")
            features, scores = load_all_data()
            df = pd.merge(features, scores, on="user", how="left")

    # -------- Remove User --------
    st.subheader("🗑 Remove User")
    features_path = os.path.join(DATA_DIR, "merged_features.csv")
    if os.path.exists(features_path):
        df_features = pd.read_csv(features_path)
        if "department" not in df_features.columns:
            df_features["department"] = "Unknown"

        dept_remove = st.selectbox("Select Department", sorted(df_features["department"].unique()))
        users_in_dept = df_features[df_features["department"] == dept_remove]["user"].tolist()

        if users_in_dept:
            user_to_remove = st.selectbox("Select User to Remove", users_in_dept)
            if st.button("Remove User"):
                df_features = df_features[df_features["user"] != user_to_remove]
                df_features.to_csv(features_path, index=False)
                st.success(f"🗑 Removed {user_to_remove} from {dept_remove}. Retraining models...")
                retrain_all()
                features, scores = load_all_data()
                df = pd.merge(features, scores, on="user", how="left")
        else:
            st.warning(f"No users in {dept_remove} department.")
    else:
        st.info("No feature file found yet. Add users first.")

# -------------------- HOW IT WORKS --------------------
with how_tab:
    st.header("How It Works — Department-Aware Adaptive Detection")
    st.markdown("""
    - Each department (Customer Care, IT Support, Finance) has its own **IsolationForest** model.
    - When you add or remove a user, the respective department model retrains automatically.
    - Scores are normalized *within each department*, so "High Risk" means "unusual compared to peers".
    - Thresholds (q90/q70) are computed dynamically for each department.
    """)
