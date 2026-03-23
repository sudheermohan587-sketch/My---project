from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import os

app = Flask(__name__)

# ── Load & preprocess data ──────────────────────────────────────────────────
def load_data():
    df = pd.read_csv('movies.csv')
    # robust RT score parsing
    def parse_rt(val):
        if isinstance(val, str):
            if '%' in val:
                return float(val.replace('%', ''))
            if '/100' in val:
                try:
                    return float(val.split('/')[0])
                except ValueError:
                    return np.nan
            try:
                return float(val)
            except ValueError:
                return np.nan
        return val
    df['RT_Score'] = df['Rotten Tomatoes'].apply(parse_rt).astype(float)
    age_map = {'all': 0, '7+': 7, '13+': 13, '16+': 16, '18+': 18}
    df['Age_Num'] = df['Age'].map(age_map).fillna(0)
    platforms = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
    df['Platform_Count'] = df[platforms].sum(axis=1)
    return df

def run_kmeans(df, k=4):
    platforms = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
    feat_cols = ['Year', 'Age_Num', 'RT_Score', 'Netflix', 'Hulu', 'Prime Video', 'Disney+', 'Platform_Count']
    X = df[feat_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df['Cluster'] = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df['Cluster'])
    return df, km, sil

df_global = load_data()
# choose K based on silhouette? default kept at 4 for now
# could later compute optimal K similar to notebook

df_clustered, km_model, sil_score = run_kmeans(df_global)

# helper to build summary dict

def generate_summary():
    platforms = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
    summary = {
        'total_movies': len(df_global),
        'netflix_count': int(df_global['Netflix'].sum()),
        'hulu_count': int(df_global['Hulu'].sum()),
        'prime_count': int(df_global['Prime Video'].sum()),
        'disney_count': int(df_global['Disney+'].sum()),
        'avg_rt': round(df_global['RT_Score'].mean(), 1),
        'clusters': int(df_clustered['Cluster'].nunique())
    }
    return summary


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    summary_path = 'static/summary.json'
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = generate_summary()
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f)

    summary['silhouette'] = round(sil_score, 4)
    return render_template('index.html', summary=summary)


@app.route('/api/stats')
def api_stats():
    platforms = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
    stats = {
        'platform_counts': {p: int(df_global[p].sum()) for p in platforms},
        'avg_rt_per_platform': {p: round(df_global[df_global[p] == 1]['RT_Score'].mean(), 1) for p in platforms},
        'year_range': [int(df_global['Year'].min()), int(df_global['Year'].max())],
        'age_distribution': df_global['Age'].value_counts().to_dict(),
        'total': len(df_global),
        'silhouette_score': round(sil_score, 4)
    }
    return jsonify(stats)


@app.route('/api/clusters')
def api_clusters():
    platforms = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
    cluster_profiles = []
    for c in range(4):
        subset = df_clustered[df_clustered['Cluster'] == c]
        cluster_profiles.append({
            'cluster': c,
            'size': len(subset),
            'avg_rt': round(subset['RT_Score'].mean(), 1),
            'avg_year': round(subset['Year'].mean(), 1),
            'platforms': {p: int(subset[p].sum()) for p in platforms}
        })
    return jsonify(cluster_profiles)


@app.route('/api/movies')
def api_movies():
    platforms = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
    sample = df_clustered[['Title', 'Year', 'Age', 'RT_Score', 'Netflix', 'Hulu', 'Prime Video', 'Disney+', 'Cluster', 'Platform_Count']].head(100)
    return jsonify(sample.to_dict(orient='records'))


@app.route('/eda')
def eda():
    return render_template('eda.html')


@app.route('/clustering')
def clustering():
    return render_template('clustering.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)