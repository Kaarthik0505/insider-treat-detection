import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
#  Core Model Training Routine
# =====================================================
def _train_and_save_models(X):
    """Train Isolation Forest, One-Class SVM, and Autoencoder, then save models."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # --- Isolation Forest ---
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(Xs)
    iso_scores = -iso.score_samples(Xs)

    # --- One-Class SVM ---
    svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    svm.fit(Xs)
    svm_scores = -svm.decision_function(Xs)

    # --- Autoencoder (proxy) ---
    auto = MLPRegressor(hidden_layer_sizes=(8, 4, 8), max_iter=1000, random_state=42)
    auto.fit(Xs, Xs)
    auto_recon = np.mean((Xs - auto.predict(Xs)) ** 2, axis=1)

    # --- Save models for later reuse ---
    joblib.dump(iso, os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    joblib.dump(svm, os.path.join(MODEL_DIR, 'oneclass_svm.pkl'))
    joblib.dump(auto, os.path.join(MODEL_DIR, 'autoencoder.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    return iso_scores, svm_scores, auto_recon


# =====================================================
#  Adaptive Retraining (Dynamic New User Addition)
# =====================================================
def retrain_with_new_user(user_id, login_freq, files_accessed, flag):
    """
    Adds a new user dynamically and retrains models with updated data.

    Parameters
    ----------
    user_id : str
        Unique ID of the new user.
    login_freq : float
        Average or normalized login frequency.
    files_accessed : float
        Average files accessed per day or week.
    flag : int
        1 if user shows suspicious behavior (red team), else 0.
    """
    features_path = os.path.join(DATA_DIR, 'merged_features.csv')
    scores_path = os.path.join(DATA_DIR, 'anomaly_scores.csv')

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"{features_path} not found")

    # --- Load existing data ---
    df = pd.read_csv(features_path)

    # --- Construct new user record (fill missing columns safely) ---
    new_user_row = {
        'user': user_id,
        'mean_login_hour': login_freq,
        'files_per_day': files_accessed,
        'usb_per_day': 0,
        'emails_per_day': 0,
        'out_of_session_access': 0,
        'degree_centrality': 0,
        'betweenness_centrality': 0,
        'keyword_flag': 0,
        'subject_len': 0,
        'sentiment': 0,
        'is_red_team': flag
    }

    new_df = pd.DataFrame([new_user_row])

    # Align with existing columns — fill any extra with mean or 0
    for c in df.columns:
        if c not in new_df.columns:
            if df[c].dtype.kind in 'biufc':
                new_df[c] = df[c].mean() if not df[c].isna().all() else 0
            else:
                new_df[c] = 'unknown' if c == 'user' else 0
    new_df = new_df[df.columns]

    # --- Append and save merged features ---
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(features_path, index=False)

    # --- Drop non-feature columns ---
    X = df.drop(columns=[c for c in ['user', 'is_red_team'] if c in df.columns], errors='ignore')

    # --- Retrain all models dynamically ---
    iso_scores, svm_scores, auto_recon = _train_and_save_models(X)

    # --- Build anomaly scores dataframe ---
    scores_df = pd.DataFrame({
        'user': df['user'],
        'is_red_team': df['is_red_team'] if 'is_red_team' in df.columns else 0,
        'isolation_forest': iso_scores,
        'oneclass_svm': svm_scores,
        'autoencoder': auto_recon
    })

    # Normalize 0–1
    scaler = MinMaxScaler()
    scores_cols = ['isolation_forest', 'oneclass_svm', 'autoencoder']
    scores_df[scores_cols] = scaler.fit_transform(scores_df[scores_cols])

    # Compute aggregated score
    scores_df['aggregated_score'] = scores_df[scores_cols].mean(axis=1)

    # Save updated scores
    scores_df.to_csv(scores_path, index=False)

    # --- Compute thresholds ---
    q90 = float(scores_df['aggregated_score'].quantile(0.90))
    q70 = float(scores_df['aggregated_score'].quantile(0.70))

    # --- Get new user’s record ---
    new_user_score = scores_df[scores_df['user'] == user_id].iloc[0].to_dict()

    # --- Return dynamic result summary ---
    return {
        "status": "success",
        "user": new_user_score['user'],
        "aggregated_score": float(new_user_score['aggregated_score']),
        "q90_threshold": q90,
        "q70_threshold": q70,
        "total_users": int(len(scores_df)),
        "risk_level": (
            "🚩 RED" if new_user_score['aggregated_score'] > q90 else
            "🟠 YELLOW" if new_user_score['aggregated_score'] > q70 else
            "🟢 GREEN"
        )
    }
