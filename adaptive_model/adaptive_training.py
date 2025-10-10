import pandas as pd
import numpy as np
import os

DATA_DIR = 'data'

def retrain_model(user_id, login_freq, files_accessed, flag):
    """
    Simulates adaptive learning by updating user data
    and recalculating simple anomaly score metrics.
    """
    try:
        features_path = os.path.join(DATA_DIR, 'merged_features.csv')
        scores_path = os.path.join(DATA_DIR, 'anomaly_scores.csv')

        # Load existing data
        features = pd.read_csv(features_path)
        scores = pd.read_csv(scores_path)

        # Create new user entry
        new_user = {
            'user': user_id,
            'mean_login_hour': np.random.uniform(8, 18),
            'mean_logout_hour': np.random.uniform(16, 22),
            'files_per_day': files_accessed,
            'usb_per_day': np.random.randint(0, 5),
            'emails_per_day': login_freq,
            'out_of_session_access': np.random.randint(0, 3),
            'degree_centrality': np.random.random(),
            'betweenness_centrality': np.random.random(),
            'keyword_flag': flag,
            'subject_len': np.random.uniform(10, 100),
            'sentiment': np.random.uniform(-1, 1)
        }

        # Add new user
        features = pd.concat([features, pd.DataFrame([new_user])], ignore_index=True)

        # Simple anomaly scoring logic (simulated)
        anomaly_score = (
            (login_freq / (features['emails_per_day'].mean() + 1)) +
            (files_accessed / (features['files_per_day'].mean() + 1)) +
            (1.5 if flag == 1 else 0)
        )

        new_score = {
            'user': user_id,
            'isolation_forest': anomaly_score * 0.8,
            'oneclass_svm': anomaly_score * 0.9,
            'autoencoder': anomaly_score,
            'is_red_team': 0
        }

        scores = pd.concat([scores, pd.DataFrame([new_score])], ignore_index=True)

        # Save updated CSVs
        features.to_csv(features_path, index=False)
        scores.to_csv(scores_path, index=False)

        return {"status": "success", "new_user": user_id, "anomaly_score": anomaly_score}

    except Exception as e:
        return {"status": "error", "message": str(e)}
