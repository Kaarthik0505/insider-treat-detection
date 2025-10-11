# dashboard/combined_dashboard.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from adaptive_model.adaptive_training import retrain_with_new_user, retrain_all
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = 'data'

st.set_page_config(layout="wide")
st.title('AI-Powered Insider Threat Detection: Department-aware Dashboard')

# ---------------- CONFIG: department-specific guidance thresholds (for UI only) ----------------
DEPT_FEATURE_THRESHOLDS = {
    "Customer Care": {
        "mean_login_hour": (6, 22),        # typical login window
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
    # fallback:
    "Unknown": {
        "mean_login_hour": (0, 23),
        "files_per_day": (0, 200),
        "usb_per_day": (0, 5),
        "emails_per_day": (0, 200)
    }
}

# -------------------- LOAD DATA --------------------
def load_all_data():
    features_path = os.path.join(DATA_DIR, 'merged_features.csv')
    scores_path = os.path.join(DATA_DIR, 'anomaly_scores.csv')

    features = pd.read_csv(features_path) if os.path.exists(features_path) else pd.DataFrame()
    scores = pd.read_csv(scores_path) if os.path.exists(scores_path) else pd.DataFrame()

    # Ensure department exists
    if features is not None and 'department' not in features.columns:
        features['department'] = 'Unknown'

    # If scores exists but department missing, try to merge department
    if not scores.empty and 'department' not in scores.columns and 'user' in scores.columns and not features.empty:
        scores = scores.merge(features[['user', 'department']], on='user', how='left')

    return features, scores


features, scores = load_all_data()
# Ensure merged_features has department col
if 'department' not in features.columns:
    features['department'] = 'Unknown'

# Merge for display
if not scores.empty and not features.empty:
    df = pd.merge(features, scores, on='user', how='left')
else:
    df = features.copy()

# -------------------- Helper: risk assignment per department --------------------
def compute_risk_levels(df_display):
    # compute aggregated_score if missing
    if 'aggregated_score' not in df_display.columns:
        # if isolation_forest exists, use it as aggregated_score
        if 'isolation_forest' in df_display.columns:
            df_display['aggregated_score'] = df_display['isolation_forest']
        else:
            df_display['aggregated_score'] = 0.0

    df_display['aggregated_score'] = df_display['aggregated_score'].fillna(0.0)

    # For each department compute department-local thresholds and assign risk
    df_display['Risk Level'] = '🟢 Low Risk'
    for dept in df_display['department'].fillna('Unknown').unique():
        mask = df_display['department'].fillna('Unknown') == dept
        sub = df_display.loc[mask, 'aggregated_score']
        if sub.empty:
            continue
        q90 = sub.quantile(0.90)
        q70 = sub.quantile(0.70)

        df_display.loc[mask & (df_display['aggregated_score'] >= q90), 'Risk Level'] = '🔴 High Risk'
        df_display.loc[mask & (df_display['aggregated_score'] < q90) & (df_display['aggregated_score'] >= q70), 'Risk Level'] = '🟡 Medium Risk'
        df_display.loc[mask & (df_display['aggregated_score'] < q70), 'Risk Level'] = '🟢 Low Risk'

    return df_display

# -------------------- TABS --------------------
anomaly_tab, user_tab, adaptive_tab, how_tab = st.tabs([
    "Anomaly Table", "User Detail", "Adaptive Learning", "How Does It Work?"
])

# --- ANOMALY TABLE ---
with anomaly_tab:
    st.header("User Anomaly Scores (department-aware)")
    departments = sorted(list(DEPT_FEATURE_THRESHOLDS.keys()))
    selected_department = st.selectbox("Filter by Department", ["All"] + departments, index=0)

    df_display = df.copy()
    if selected_department != "All":
        df_display = df_display[df_display['department'] == selected_department].copy()

    # compute risk levels per department
    df_display = compute_risk_levels(df_display)

    # ensure showing the four features
    feature_cols = ['mean_login_hour', 'files_per_day', 'usb_per_day', 'emails_per_day']
    show_cols = ['user', 'department', 'aggregated_score', 'Risk Level'] + [c for c in feature_cols if c in df_display.columns]
    st.dataframe(df_display[show_cols].fillna(''), height=480)

    st.subheader("Top 5 (by aggregated_score)")
    top5 = df_display.sort_values('aggregated_score', ascending=False).head(5)
    if not top5.empty and 'aggregated_score' in top5.columns:
        st.bar_chart(top5.set_index('user')['aggregated_score'])

# --- USER DETAIL ---
with user_tab:
    st.header("User Detail")
    if df.empty:
        st.info("No users available. Add users in Adaptive Learning tab.")
    else:
        selected_user = st.selectbox("Select User", df['user'].tolist())
        row = df[df['user'] == selected_user].iloc[0]
        st.markdown(f"**Department:** {row.get('department','Unknown')}")
        st.markdown(f"**Aggregated score:** {row.get('aggregated_score', 0.0):.3f}")
        st.markdown(f"**Risk Level:** {compute_risk_levels(pd.DataFrame([row]))['Risk Level'].iloc[0]}")
        st.write("**Features:**")
        features_shown = {k: row[k] for k in ['mean_login_hour', 'files_per_day', 'usb_per_day', 'emails_per_day'] if k in row}
        st.json(features_shown)
        st.write("**Anomaly Scores:**")
        st.json({'isolation_forest': row.get('isolation_forest', None)})

# --- ADAPTIVE LEARNING ---
with adaptive_tab:
    st.header("Adaptive Learning — Add / Remove Users (department-aware)")

    st.subheader("➕ Add New User")
    with st.form("add_user_form"):
        new_user = {}
        new_user['user'] = st.text_input("User ID (unique)", value="")
        dept = st.selectbox("Department", sorted(DEPT_FEATURE_THRESHOLDS.keys()), index=0)
        new_user['department'] = dept

        # display guidance thresholds for the selected department
        thr = DEPT_FEATURE_THRESHOLDS.get(dept, DEPT_FEATURE_THRESHOLDS['Unknown'])
        st.markdown(f"**Guidance for {dept}:**")
        st.markdown(f"- Typical login hour range: {thr['mean_login_hour']}")
        st.markdown(f"- Typical files/day range: {thr['files_per_day']}")
        st.markdown(f"- Typical usb/day range: {thr['usb_per_day']}")
        st.markdown(f"- Typical emails/day range: {thr['emails_per_day']}")

        new_user['mean_login_hour'] = float(st.number_input("Average Login Hour (0-23)", min_value=0.0, max_value=23.0, value=float((thr['mean_login_hour'][0]+thr['mean_login_hour'][1])/2)))
        new_user['files_per_day'] = int(st.number_input("Files Accessed Per Day", min_value=0, max_value=10000, value=int((thr['files_per_day'][0]+thr['files_per_day'][1])//2)))
        new_user['usb_per_day'] = float(st.number_input("USB Devices Per Day", min_value=0.0, max_value=100.0, value=float((thr['usb_per_day'][0]+thr['usb_per_day'][1])/2)))
        new_user['emails_per_day'] = int(st.number_input("Emails Sent Per Day", min_value=0, max_value=10000, value=int((thr['emails_per_day'][0]+thr['emails_per_day'][1])//2)))

        submitted = st.form_submit_button("Add user & retrain")

    if submitted:
        if not new_user['user'] or new_user['user'].strip() == "":
            st.error("Please enter a valid user id.")
        else:
            st.info("Adding new user data and retraining model...")

            # Retrain the model with the new user
            result = retrain_with_new_user({
                "user": user_id,
                "mean_login_hour": login_freq,
                "files_per_day": files_accessed,
                "usb_per_day": usb_count,
                "emails_per_day": email_count,
            })

            # Reload updated anomaly scores
            updated_scores = pd.read_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'))

            # Compute high risk threshold (90th percentile)
            high_risk_threshold = updated_scores['isolation_forest'].quantile(0.90)

            # Get the new user's anomaly score
            new_user_score = updated_scores.loc[
                updated_scores['user'] == user_id, 'isolation_forest'
            ].values[0]

            # 🚨 Popup alert for HIGH RISK user
            if new_user_score >= high_risk_threshold:
                st.toast(
                    f"🚨 ALERT: New user `{user_id}` is HIGH RISK (🔴)! Immediate review recommended.",
                    icon="⚠️",
                    duration=5000
                )

            st.success(f"✅ Model retrained successfully with new user `{user_id}`!")
            st.json(result)
            st.rerun()


    # Remove User
    st.subheader("🗑 Remove Existing User")
    try:
        users_list = pd.read_csv(os.path.join(DATA_DIR, "merged_features.csv"))["user"].tolist()
        user_to_remove = st.selectbox("Select User to Remove", users_list)
        if st.button("🚫 Remove Selected User"):
            features_df = pd.read_csv(os.path.join(DATA_DIR, "merged_features.csv"))
            scores_df = pd.read_csv(os.path.join(DATA_DIR, "anomaly_scores.csv"))

            if user_to_remove in features_df["user"].values:
                features_df = features_df[features_df["user"] != user_to_remove]
                features_df.to_csv(os.path.join(DATA_DIR, "merged_features.csv"), index=False)
            if user_to_remove in scores_df["user"].values:
                scores_df = scores_df[scores_df["user"] != user_to_remove]
                scores_df.to_csv(os.path.join(DATA_DIR, "anomaly_scores.csv"), index=False)

            st.success(f"🗑 User `{user_to_remove}` removed successfully. Retraining model...")

            if not features_df.empty:
                from adaptive_model.adaptive_training import _train_and_save_models
                import numpy as np
                from sklearn.preprocessing import MinMaxScaler

                X = features_df.drop(columns=[c for c in ['user'] if c in features_df.columns], errors='ignore')
                iso_scores, _, _ = _train_and_save_models(X)  # only use isolation forest returned scores

                scores_df = pd.DataFrame({
                    'user': features_df['user'],
                    'isolation_forest': iso_scores
                })

                scaler = MinMaxScaler()
                score_cols = ['isolation_forest']
                scores_df[score_cols] = scaler.fit_transform(scores_df[score_cols])
                scores_df['aggregated_score'] = scores_df[score_cols].mean(axis=1)
                scores_df.to_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'), index=False)

                st.success("✅ Model retrained successfully after deletion.")
            st.rerun()

    except Exception as e:
        st.error(f"Error while removing user: {e}")

# --- HOW TAB ---
with how_tab:
    st.header("How it works (department-aware)")
    st.markdown("""
    - Each user belongs to a department (Customer Care, IT Support, Finance, etc.)
    - We train a small IsolationForest per department on that department's feature set.
    - Scores are normalized **within each department** so comparisons are relative to the department norms.
    - Adaptive add/remove updates the merged user feature file and re-trains department models.
    """)

