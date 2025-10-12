import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_FILE = os.path.join(DATA_DIR, "merged_features.csv")
SCORES_FILE = os.path.join(DATA_DIR, "anomaly_scores.csv")

# --------------------------------------------
# Helper: Load data safely
# --------------------------------------------
def load_features():
    if os.path.exists(FEATURE_FILE):
        df = pd.read_csv(FEATURE_FILE)
    else:
        df = pd.DataFrame(columns=[
            "user", "department",
            "mean_login_hour", "files_per_day", "usb_per_day", "emails_per_day"
        ])
    if "department" not in df.columns:
        df["department"] = "Unknown"
    return df


# --------------------------------------------
# Helper: Train model per department
# --------------------------------------------
def train_department_models(df):
    """
    Trains Isolation Forest per department and returns anomaly scores.
    """
    all_scores = []

    # Loop through departments
    for dept, sub_df in df.groupby("department"):
        if len(sub_df) < 2:
            # not enough data, skip training
            for _, row in sub_df.iterrows():
                all_scores.append({
                    "user": row["user"],
                    "department": dept,
                    "isolation_forest": 0.0,
                    "aggregated_score": 0.0
                })
            continue

        # Features for training
        X = sub_df[["mean_login_hour", "files_per_day", "usb_per_day", "emails_per_day"]].fillna(0)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X_scaled)
        scores = -model.decision_function(X_scaled)  # higher = more anomalous

        # Normalize scores (0–1)
        scaled_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

        # Save model and scaler
        model_path = os.path.join(MODEL_DIR, f"{dept.replace(' ', '_')}_iforest.joblib")
        scaler_path = os.path.join(MODEL_DIR, f"{dept.replace(' ', '_')}_scaler.joblib")
        dump(model, model_path)
        dump(scaler, scaler_path)

        # Store scores
        for i, row in enumerate(sub_df.itertuples(index=False)):
            all_scores.append({
                "user": row.user,
                "department": dept,
                "isolation_forest": float(scaled_scores[i]),
                "aggregated_score": float(scaled_scores[i])
            })

    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(SCORES_FILE, index=False)
    return scores_df


# --------------------------------------------
# Add new user and retrain their department model
# --------------------------------------------
def retrain_with_new_user(new_user_dict):
    df = load_features()

    user_id = new_user_dict["user"]
    dept = new_user_dict.get("department", "Unknown")

    # Remove old entry if exists
    df = df[df["user"] != user_id]

    # Add new user
    df = pd.concat([df, pd.DataFrame([new_user_dict])], ignore_index=True)
    df.to_csv(FEATURE_FILE, index=False)

    # Retrain only for that department
    sub_df = df[df["department"] == dept]
    scores_df = train_department_models(df)

    # Calculate department risk thresholds
    dept_scores = scores_df[scores_df["department"] == dept]["aggregated_score"]
    q90 = dept_scores.quantile(0.9) if not dept_scores.empty else 0.8
    q70 = dept_scores.quantile(0.7) if not dept_scores.empty else 0.5

    user_score = scores_df.loc[scores_df["user"] == user_id, "aggregated_score"].values
    user_score = float(user_score[0]) if len(user_score) > 0 else 0.0

    return {
        "user": user_id,
        "department": dept,
        "aggregated_score": user_score,
        "q90": q90,
        "q70": q70
    }


# --------------------------------------------
# Retrain all departments
# --------------------------------------------
def retrain_all():
    df = load_features()
    if df.empty:
        return {"status": "No data to train"}

    scores_df = train_department_models(df)
    return {"status": "Retrained all", "total_users": len(df)}


if __name__ == "__main__":
    print("Adaptive training script — retraining all departments...")
    summary = retrain_all()
    print(summary)
