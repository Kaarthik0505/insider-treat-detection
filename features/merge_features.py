import pandas as pd
import os

DATA_DIR = 'data'

# Load features
df_classic = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'))
df_nlp = pd.read_csv(os.path.join(DATA_DIR, 'nlp_email_features.csv'))

# Aggregate NLP features per user (mean)
df_nlp['user'] = df_nlp['sender'].str.replace('@company.com', '', regex=False)
df_nlp_agg = df_nlp.groupby('user').agg({
    'keyword_flag': 'mean',
    'subject_len': 'mean',
    'sentiment': 'mean'
}).reset_index()

# Merge classic + NLP features
df = df_classic.merge(df_nlp_agg, on='user', how='left')

# Save merged file
output_path = os.path.join(DATA_DIR, 'merged_features.csv')
df.to_csv(output_path, index=False)

print(f'Merged features saved to {output_path}')
