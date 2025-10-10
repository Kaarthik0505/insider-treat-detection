import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
import os

DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load features
df = pd.read_csv(os.path.join(DATA_DIR, 'merged_features.csv'))
X = df.drop(['user', 'is_red_team'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Isolation Forest
iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X_scaled)
iso_scores = -iso.score_samples(X_scaled)    # keeps previous behaviour

joblib.dump(iso, os.path.join(MODEL_DIR, 'isolation_forest.pkl'))

# One-Class SVM
svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
svm.fit(X_scaled)
svm_scores = -svm.decision_function(X_scaled)
joblib.dump(svm, os.path.join(MODEL_DIR, 'oneclass_svm.pkl'))

# Autoencoder (MLPRegressor as proxy)
auto = MLPRegressor(hidden_layer_sizes=(8,4,8), max_iter=1000, random_state=42)
auto.fit(X_scaled, X_scaled)
auto_recon = np.mean((X_scaled - auto.predict(X_scaled))**2, axis=1)
joblib.dump(auto, os.path.join(MODEL_DIR, 'autoencoder.pkl'))

# ------------------------------------------
# NORMALIZE SCORES to 0-1 (so scales are consistent)
from sklearn.preprocessing import MinMaxScaler
scores_df = pd.DataFrame({
    'user': df['user'],
    'is_red_team': df['is_red_team'],
    'isolation_forest': iso_scores,
    'oneclass_svm': svm_scores,
    'autoencoder': auto_recon
})

scaler = MinMaxScaler()
scores_cols = ['isolation_forest', 'oneclass_svm', 'autoencoder']
scores_df[scores_cols] = scaler.fit_transform(scores_df[scores_cols])

# Create aggregated score as the mean of normalized methods (you can weight if desired)
scores_df['aggregated_score'] = scores_df[scores_cols].mean(axis=1)

# Save normalized scores
scores_df.to_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'), index=False)
print('Models trained, normalized scores saved to data/anomaly_scores.csv')
