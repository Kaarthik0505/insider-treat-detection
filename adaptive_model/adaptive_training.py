# adaptive_model/adaptive_training.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


def _ensure_department_column(df):
    if 'department' not in df.columns:
        df['department'] = 'Unknown'
    df['department'] = df['department'].fillna('Unknown')
    return df


def _train_isolation_per_department(df):
    """
    For each department present in df:
      - train an IsolationForest on that department's feature rows,
      - compute raw scores, MinMax-normalize per-department,
    Returns combined scores_df with columns: user, department, isolation_forest, aggregated_score
    and saves per-department models/scalers as isolation_forest__<dept>.pkl and scaler__<dept>.pkl
    """
    df = _ensure_department_column(df)
    results = []

    for dept in sorted(df['department'].unique()):
        df_dept = df[df['department'] == dept].reset_index(drop=True)
        if df_dept.empty:
            continue

        # Features used for training: drop 'user' and 'department'
        X = df_dept.drop(columns=[c for c in ['user', 'department'] if c in df_dept.columns], errors='ignore')
        if X.shape[0] < 2 or X.shape[1] == 0:
            # Not enough data to train properly - produce zero scores
            iso_scores = np.zeros(len(df_dept))
            scaler = None
            iso_model = None
        else:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            iso_model = IsolationForest(contamination=0.1, random_state=42)
            iso_model.fit(Xs)
            iso_scores = -iso_model.score_samples(Xs)

            # Save model and scaler for this department (safe file names)
            safe_dept = dept.replace(' ', '_').lower()
            joblib.dump(iso_model, os.path.join(MODEL_DIR, f'isolation_forest__{safe_dept}.pkl'))
            joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler__{safe_dept}.pkl'))

        # Normalize iso_scores within department
        if len(iso_scores) > 0:
            mm = MinMaxScaler()
            iso_norm = mm.fit_transform(np.array(iso_scores).reshape(-1, 1)).flatten()
        else:
            iso_norm = np.array([])

        for i, user in enumerate(df_dept['user'].tolist()):
            results.append({
                'user': user,
                'department': dept,
                'isolation_forest': float(iso_norm[i]) if len(iso_norm) > i else 0.0
            })

    scores_df = pd.DataFrame(results)
    if scores_df.empty:
        return pd.DataFrame(columns=['user', 'department', 'isolation_forest', 'aggregated_score'])

    # aggregated_score is same as isolation_forest (single model) - kept for compatibility
    scores_df['aggregated_score'] = scores_df['isolation_forest']
    # Save global anomaly_scores.csv
    scores_df.to_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'), index=False)

    return scores_df


def retrain_with_new_user(new_user: dict):
    """
    Add a new user row (dictionary must contain 'user' and 'department' keys, plus feature fields)
    Append to merged_features.csv, retrain per-department models and update anomaly_scores.csv
    Returns a summary dict with new user's score and department-level thresholds (q90, q70).
    """
    features_path = os.path.join(DATA_DIR, 'merged_features.csv')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"{features_path} not found")

    df = pd.read_csv(features_path)
    df = _ensure_department_column(df)

    # Build new_df - ensure compatible columns
    new_df = pd.DataFrame([new_user])
    # If missing columns in new_df, fill numeric columns with department mean or 0, else 'unknown'
    for c in df.columns:
        if c not in new_df.columns:
            if df[c].dtype.kind in 'biufc':  # numeric
                new_df[c] = df[c].mean() if not df[c].isna().all() else 0
            else:
                new_df[c] = 'Unknown' if c == 'department' else ('unknown' if c == 'user' else '')

    new_df = new_df[df.columns]  # align column order
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(features_path, index=False)

    # Retrain per department and save scores
    scores_df = _train_isolation_per_department(df)

    # compute department thresholds (q90/q70) and return new user info
    dept = new_user.get('department', 'Unknown')
    dept_scores = scores_df[scores_df['department'] == dept]
    if dept_scores.empty:
        return {
            "user": new_user.get('user'),
            "aggregated_score": 0.0,
            "q90": 0.0,
            "q70": 0.0,
            "total_users": int(len(scores_df))
        }

    q90 = float(dept_scores['aggregated_score'].quantile(0.90))
    q70 = float(dept_scores['aggregated_score'].quantile(0.70))
    new_row = dept_scores[dept_scores['user'] == new_user['user']].iloc[0].to_dict()

    return {
        "user": new_row['user'],
        "aggregated_score": float(new_row['aggregated_score']),
        "q90": q90,
        "q70": q70,
        "total_users": int(len(scores_df))
    }


def retrain_all():
    """
    Retrain models using the current merged_features.csv contents.
    Useful after deletion.
    Returns a summary dict: number of users and departments trained.
    """
    features_path = os.path.join(DATA_DIR, 'merged_features.csv')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"{features_path} not found")

    df = pd.read_csv(features_path)
    df = _ensure_department_column(df)

    scores_df = _train_isolation_per_department(df)

    return {
        "total_users": int(len(scores_df)),
        "departments": sorted(df['department'].unique())
    }
