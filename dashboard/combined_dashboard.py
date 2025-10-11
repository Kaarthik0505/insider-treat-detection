import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from adaptive_model.adaptive_training import retrain_with_new_user

DATA_DIR = 'data'

st.set_page_config(layout="wide")
st.title('AI-Powered Insider Threat Detection: Combined Dashboard')


# -------------------- LOAD DATA --------------------
def load_all_data():
    features = pd.read_csv(os.path.join(DATA_DIR, 'merged_features.csv'))
    scores = pd.read_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'))
    file_access = pd.read_csv(os.path.join(DATA_DIR, 'file_access.csv'), parse_dates=['access_time'])
    usb_usage = pd.read_csv(os.path.join(DATA_DIR, 'usb_usage.csv'), parse_dates=['plug_time', 'unplug_time'])
    return features, scores, file_access, usb_usage


features, scores, file_access, usb_usage = load_all_data()
df = pd.merge(features, scores, on='user')


# -------------------- ANOMALY HELPERS --------------------
def assign_risk_band(score, high_risk_threshold, medium_risk_threshold):
    if score >= high_risk_threshold:
        return '🔴 High Risk'
    elif score >= medium_risk_threshold:
        return '🟡 Medium Risk'
    else:
        return '🟢 Low Risk'


# -------------------- DASHBOARD TABS --------------------
anomaly_tab, user_tab, adaptive_tab = st.tabs([
    "Anomaly Table", "User Detail", "Adaptive Learning"
])


# === ANOMALY TABLE ===
with anomaly_tab:
    st.header('User Anomaly Scores')
    score_method = 'isolation_forest'  # Fixed to isolation_forest only
    df['rank'] = df[score_method].rank(ascending=False)
    df_sorted = df.sort_values(score_method, ascending=False)
    df_sorted['score'] = df_sorted[score_method]
    high_risk_threshold = df_sorted['score'].quantile(0.90)
    medium_risk_threshold = df_sorted['score'].quantile(0.70)
    df_sorted['Risk Level'] = df_sorted['score'].apply(
        lambda score: assign_risk_band(score, high_risk_threshold, medium_risk_threshold)
    )
    st.dataframe(df_sorted[['user', 'Risk Level', score_method, 'rank']], height=500)
    st.subheader('Top 5 High-Risk Users')
    st.bar_chart(df_sorted.head(5).set_index('user')[score_method])


# === USER DETAIL TAB ===
with user_tab:
    st.header('User Detail')
    selected_user = st.selectbox('Select User', df_sorted['user'])
    user_row = df_sorted[df_sorted['user'] == selected_user].iloc[0]
    st.write('**Risk Level:**', user_row['Risk Level'])
    st.write('**Features:**')
    st.json({k: user_row[k] for k in [
        'mean_login_hour', 'mean_logout_hour', 'files_per_day', 'usb_per_day',
        'emails_per_day', 'out_of_session_access', 'degree_centrality',
        'betweenness_centrality', 'keyword_flag', 'subject_len', 'sentiment'
    ] if k in user_row})
    st.write('**Anomaly Score (Isolation Forest):**')
    st.json({'isolation_forest': user_row['isolation_forest']})


# === ADAPTIVE LEARNING TAB ===
with adaptive_tab:
    st.header("🧠 Adaptive Learning Module")

    # Add User
    st.subheader("➕ Add New User Data")
    user_id = st.text_input("Enter User ID (e.g., user_101)")
    login_freq = st.number_input("Average Login Hour", min_value=0.0, max_value=24.0, value=9.0)
    files_accessed = st.number_input("Files Accessed Per Day", min_value=0, max_value=1000, value=20)
    usb_count = st.number_input("USB Devices Used Per Day", min_value=0, max_value=50, value=0)
    email_count = st.number_input("Emails Sent Per Day", min_value=0, max_value=500, value=0)

    if st.button("📥 Add User and Retrain Model"):
        if not user_id.strip():
            st.error("❌ Please enter a valid user ID.")
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
        st.error(f"Error loading users: {e}")
