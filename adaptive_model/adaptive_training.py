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

# --- helper: normalize input for retrain_with_new_user ---
def _normalize_retrain_args(*args, **kwargs):
    """
    Accepts either:
      retrain_with_new_user(new_user_dict)
    or
      retrain_with_new_user(user_id, login_freq, files_accessed, flag)
    Returns a normalized dict.
    """
    if len(args) == 1 and isinstance(args[0], dict):
        return args[0]

    d = {}
    if len(args) >= 1:
        d["user"] = args[0]
    if len(args) >= 2:
        d["mean_login_hour"] = args[1]
    if len(args) >= 3:
        d["files_per_day"] = args[2]
    if len(args) >= 4:
        d["is_red_team"] = args[3]

    for k, v in kwargs.items():
        d[k] = v

    return d


def _train_and_save_models(X):
    """Train Isolation Forest, One-Class SVM, and Autoencoder; save models."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(Xs)
    iso_scores = -iso.score_samples(Xs)

    svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    svm.fit(Xs)
    svm_scores = -svm.decision_function(Xs)

    auto = MLPRegressor(hidden_layer_sizes=(8, 4, 8), max_iter=1000, random_state=42)
    auto.fit(Xs, Xs)
    auto_recon = np.mean((Xs - auto.predict(Xs)) ** 2, axis=1)

    joblib.dump(iso, os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    joblib.dump(svm, os.path.join(MODEL_DIR, 'oneclass_svm.pkl'))
    joblib.dump(auto, os.path.join(MODEL_DIR, 'autoencoder.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    return iso_scores, svm_scores, auto_recon


def retrain_with_new_user(*args, **kwargs):
    """
    Dynamically retrain model when a new user is added or removed.
    Accepts either a dict or args (user_id, login_freq, files_accessed, flag).
    """
    new_user_row = _normalize_retrain_args(*args, **kwargs)

    features_path = os.path.join(DATA_DIR, 'merged_features.csv')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"{features_path} not found")

    df = pd.read_csv(features_path)
    new_df = pd.DataFrame([new_user_row])

    for c in df.columns:
        if c not in new_df.columns:
            if df[c].dtype.kind in 'biufc':
                new_df[c] = df[c].mean() if not df[c].isna().all() else 0
            else:
                new_df[c] = 'unknown' if c == 'user' else 0

    new_df = new_df[df.columns]
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(features_path, index=False)

    X = df.drop(columns=[c for c in ['user', 'is_red_team'] if c in df.columns], errors='ignore').copy()

    iso_scores, svm_scores, auto_recon = _train_and_save_models(X)

    scores_df = pd.DataFrame({
        'user': df['user'],
        'is_red_team': df['is_red_team'] if 'is_red_team' in df.columns else 0,
        'isolation_forest': iso_scores,
        'oneclass_svm': svm_scores,
        'autoencoder': auto_recon
    })

    scaler = MinMaxScaler()
    score_cols = ['isolation_forest', 'oneclass_svm', 'autoencoder']
    scores_df[score_cols] = scaler.fit_transform(scores_df[score_cols])
    scores_df['aggregated_score'] = scores_df[score_cols].mean(axis=1)
    scores_df.to_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'), index=False)

    q90 = float(scores_df['aggregated_score'].quantile(0.90))
    q70 = float(scores_df['aggregated_score'].quantile(0.70))

    new_row = scores_df[scores_df['user'] == new_user_row['user']].iloc[0].to_dict()

    # --- Return dynamic result summary ---
    return {
        "user": new_row['user'],
        "aggregated_score": float(new_row['aggregated_score']),
        "q90": q90,
        "q70": q70,
        "total_users": len(scores_df)
    }

