import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
from adaptive_model.adaptive_training import retrain_with_new_user


DATA_DIR = 'data'

st.set_page_config(layout="wide")
st.title('AI-Powered Insider Threat Detection: Combined Dashboard')

# Load data
def load_all_data():
    features = pd.read_csv(os.path.join(DATA_DIR, 'merged_features.csv'))
    scores = pd.read_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'))
    file_access = pd.read_csv(os.path.join(DATA_DIR, 'file_access.csv'), parse_dates=['access_time'])
    usb_usage = pd.read_csv(os.path.join(DATA_DIR, 'usb_usage.csv'), parse_dates=['plug_time', 'unplug_time'])
    return features, scores, file_access, usb_usage

features, scores, file_access, usb_usage = load_all_data()
df = pd.merge(features, scores, on='user')

# Prepare node attributes for graph
def get_node_attrs():
    attrs = {}
    for _, row in scores.iterrows():
        anomaly = max(row['isolation_forest'], row['oneclass_svm'], row['autoencoder'])
        red_team = row['is_red_team']
        attrs[row['user']] = {
            'anomaly': anomaly,
            'red_team': red_team,
            'high_risk': (anomaly > 1.0) or (red_team == 1)
        }
    return attrs
attrs = get_node_attrs()

# Build full graph
def build_graph():
    G = nx.Graph()
    for _, row in file_access.iterrows():
        G.add_edge(row['user'], row['file'], type='access')
    for _, row in usb_usage.iterrows():
        G.add_edge(row['user'], row['device'], type='usb')
    return G
G = build_graph()

# At-risk subgraph
def get_at_risk_subgraph(G, attrs):
    high_risk_nodes = {n for n, v in attrs.items() if v['high_risk']}
    connected_nodes = set()

    for node in high_risk_nodes:
        if node not in G:
            # Skip users who are not yet in the network
            continue
        connected_nodes.add(node)
        connected_nodes.update(G.neighbors(node))

    # If no valid nodes found, return an empty graph
    if not connected_nodes:
        return nx.Graph()

    return G.subgraph(connected_nodes).copy()


# === ✅ UPDATED TABS (Added Adaptive Learning Tab) ===
anomaly_tab, user_tab, graph_tab, adaptive_tab, how_tab = st.tabs([
    "Anomaly Table", "User Detail", "At-Risk Graph", "Adaptive Learning", "How Does It Work?"
])

with anomaly_tab:
    st.header('User Anomaly Scores')
    score_method = st.selectbox('Select Model', ['isolation_forest', 'oneclass_svm', 'autoencoder'], key='score_method')
    df['Red Team'] = df['is_red_team_x'].apply(lambda x: '🚩' if x == 1 else '') if 'is_red_team_x' in df.columns else df['is_red_team'].apply(lambda x: '🚩' if x == 1 else '')
    df['rank'] = df[score_method].rank(ascending=False)
    df_sorted = df.sort_values(score_method, ascending=False)

    # === 🧠 Dynamic Risk Banding Logic ===
    df_sorted['score'] = df_sorted[score_method]
    high_risk_threshold = df_sorted['score'].quantile(0.90)
    medium_risk_threshold = df_sorted['score'].quantile(0.70)

    def assign_risk_band(score):
        if score >= high_risk_threshold:
            return '🔴 High Risk'
        elif score >= medium_risk_threshold:
            return '🟡 Medium Risk'
        else:
            return '🟢 Low Risk'

    df_sorted['Risk Level'] = df_sorted['score'].apply(assign_risk_band)
    # === End Dynamic Logic ===

    cols = ['user', 'Red Team', 'Risk Level', score_method, 'rank'] + [c for c in df.columns if c not in ['user', score_method, 'rank', 'Red Team', 'Risk Level']]
    st.dataframe(df_sorted[cols], height=500)

    st.subheader('Top 5 High-Risk Users')
    top5 = df_sorted.head(5)
    st.bar_chart(top5.set_index('user')[score_method])

with user_tab:
    st.header('User Detail')
    selected_user = st.selectbox('Select User', df_sorted['user'], key='user_detail')
    user_row = df_sorted[df_sorted['user'] == selected_user].iloc[0]
    st.write('**Red Team:**', '🚩' if user_row['Red Team'] else 'No')
    st.write('**Risk Level:**', user_row['Risk Level'])
    st.write('**Features:**')
    st.json({k: user_row[k] for k in ['mean_login_hour', 'mean_logout_hour', 'files_per_day', 'usb_per_day', 'emails_per_day', 'out_of_session_access', 'degree_centrality', 'betweenness_centrality', 'keyword_flag', 'subject_len', 'sentiment'] if k in user_row})
    st.write('**Anomaly Scores:**')
    st.json({k: user_row[k] for k in ['isolation_forest', 'oneclass_svm', 'autoencoder']})

with graph_tab:
    st.header('At-Risk Nodes and Their Connections')

    try:
        subG = get_at_risk_subgraph(G, attrs)
    except nx.NetworkXError as e:
        st.warning("⚠️ Some new users are not yet part of the graph. Rebuilding network...")
        G = build_graph()  # rebuilds the full graph
        subG = get_at_risk_subgraph(G, attrs)

    net = Network(height='900px', width='100%', notebook=False, bgcolor='#222222', font_color='white')

    net.barnes_hut(gravity=-2000, central_gravity=0.1, spring_length=200, spring_strength=0.01, damping=0.85, overlap=1)
    net.set_options('''
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "fit": true, "iterations": 2500, "updateInterval": 50},
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.01,
          "damping": 0.85,
          "avoidOverlap": 1
        }
      }
    }
    ''')
    for node in subG.nodes():
        if node in attrs:
            score = attrs[node]['anomaly']
            red = attrs[node]['red_team']
            color = 'red' if red else ('orange' if score > 1.5 else 'yellow' if score > 1.0 else 'lightblue')
            size = 30 if red else (20 if score > 1.5 else 15 if score > 1.0 else 10)
            title = f"User: {node}<br>Anomaly Score: {score:.2f}<br>Red Team: {'Yes' if red else 'No'}"
        elif str(node).startswith('file'):
            color = 'green'
            size = 8
            title = f"File: {node}"
        elif str(node).startswith('usb'):
            color = 'purple'
            size = 8
            title = f"Device: {node}"
        else:
            color = 'gray'
            size = 8
            title = str(node)
        net.add_node(node, label=str(node), color=color, size=size, title=title)
    for edge in subG.edges(data=True):
        net.add_edge(edge[0], edge[1], color='gray' if edge[2]['type']=='access' else 'purple')
    net.save_graph('dashboard/graph.html')
    st.components.v1.html(open('dashboard/graph.html', 'r', encoding='utf-8').read(), height=900, scrolling=False)

# === 🧠 ADAPTIVE LEARNING TAB START ===
from adaptive_model.adaptive_training import retrain_model

with adaptive_tab:
    st.header("🧠 Adaptive Learning Module")

    st.write("### Add New User Data (Judge Simulation)")
    user_id = st.text_input("Enter User ID (e.g., user_101)")
    login_freq = st.number_input("Login Frequency", min_value=0, max_value=100, value=10)
    files_accessed = st.number_input("Files Accessed", min_value=0, max_value=500, value=20)
    suspicious_activity = st.selectbox("Suspicious Behavior?", ["No", "Yes"])

    flag = 1 if suspicious_activity == "Yes" else 0

    if st.button("📥 Add User and Retrain Model"):
        st.info("Adding new user data and retraining model...")
        result = retrain_model(user_id, login_freq, files_accessed, flag)
        st.success("✅ Model retrained successfully!")

        st.write("### 🔍 Updated Model Performance:")
        st.json(result)
    if st.button("📥 Add User and Retrain Model"):
        st.info("Adding new user data and retraining model...")
        result = retrain_with_new_user({
    "user": user_id,
    "mean_login_hour": login_freq,
    "files_per_day": files_accessed,
    "usb_per_day": 0,
    "emails_per_day": 0,
    "is_red_team": flag
})


    st.success(f"✅ Model retrained successfully with new user `{user_id}`!")

    st.write("### 🔍 Updated Model Performance:")
    st.json(result)

# === 🧠 ADAPTIVE LEARNING TAB END ===

with how_tab:
    st.header('How Does It Work?')
    st.markdown('''


## System Overview
This system detects insider threats by analyzing user behavior, system access, and relationships using advanced machine learning and graph analysis techniques.

---

### 1. **Data Simulation & Feature Engineering**
- **Simulated Logs:** The system generates synthetic logs for user logins, file access, USB usage, and emails, mimicking real organizational activity.
- **Feature Engineering:** Extracts features such as:
    - Login/logout patterns (mean hours, frequency)
    - File/USB/email activity rates
    - Out-of-session file access
    - Graph centrality (degree, betweenness)
    - NLP features from email subjects (keyword flags, length)

---

### 2. **Anomaly Detection Algorithms**
- **Isolation Forest**
    - *Mathematics:* Randomly partitions data to isolate points. Anomalies are isolated faster (shorter average path length in trees).
    - *Computer Science:* Ensemble of binary trees; each tree splits on random features/values. The anomaly score is based on the average path length to isolate a sample.
- **One-Class SVM**
    - *Mathematics:* Finds a boundary in feature space that encloses most data (support vectors). Points outside are anomalies.
    - *Computer Science:* Uses kernel methods (e.g., RBF) to map data to high-dimensional space and find a maximal margin hyperplane.
- **Autoencoder**
    - *Mathematics:* Neural network learns to compress and reconstruct input. High reconstruction error indicates anomaly.
    - *Computer Science:* Trains a feedforward neural network (MLP) to minimize reconstruction loss (MSE) between input and output.

---

### 3. **Graph Analysis**
- **Entity Graph:** Users, files, and devices are nodes; edges represent access or usage.
- **Centrality Measures:**
    - *Degree Centrality:* Number of connections (activity level).
    - *Betweenness Centrality:* Frequency a node lies on shortest paths (potential for information flow/control).
- **At-Risk Subgraph:** Focuses on high-risk users and their direct connections for visualization and investigation.

---

### 4. **Explainability**
- **SHAP (SHapley Additive exPlanations):**
    - *Mathematics:* Based on cooperative game theory; attributes model output to each feature by averaging over all possible feature orderings.
    - *Computer Science:* Computes feature importances for each prediction, helping analysts understand why a user is flagged.
- **LIME (Local Interpretable Model-agnostic Explanations):**
    - *Mathematics:* Fits a simple, interpretable model locally around a prediction to approximate the complex model.
    - *Computer Science:* Perturbs input data and observes output changes to estimate feature influence (not supported for Isolation Forest, but available for other models).

---

### 5. **Dashboard & Visualization**
- **Streamlit:** Interactive web app for data exploration, anomaly review, and graph visualization.
- **PyVis/NetworkX:** Renders interactive network graphs for at-risk nodes and their relationships.

---

### 6. **Red Team Simulation**
- Injects malicious behaviors (after-hours access, mass downloads, suspicious USB usage) to test detection capability.

---

## Summary
This system combines unsupervised machine learning, graph theory, and explainable AI to provide a robust, interpretable approach to insider threat detection.
''') 