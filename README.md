# 🎬 Best Streaming Service Analysis
## EDA + K-Means Clustering | Flask Web App

### Project Structure
```
streaming_project/
├── app.py                    # Flask backend
├── movies.csv                # Dataset (generated / upload Kaggle CSV)
├── movies_clustered.csv      # Dataset with cluster labels
├── requirements.txt
├── notebooks/
│   └── streaming_analysis.ipynb   # Full EDA + ML notebook
├── templates/
│   ├── index.html            # Landing page
│   ├── eda.html              # EDA visualizations
│   └── clustering.html       # K-Means results
└── static/
    ├── platform_dist.png
    ├── rt_distribution.png
    ├── age_dist.png
    ├── year_vs_rt.png
    ├── correlation.png
    ├── elbow.png
    ├── clusters.png
    └── cluster_platforms.png
```

### Dataset
- Source: https://www.kaggle.com/datasets/ruchi798/movies-on-netflix-prime-video-hulu-and-disney
- Columns: Title, Year, Age, Rotten Tomatoes, Netflix, Hulu, Prime Video, Disney+

### Algorithm: K-Means Clustering
- **K = 4** clusters selected via Elbow Method + Silhouette Score
- Features: Year, Age_Num, RT_Score, Netflix, Hulu, Prime Video, Disney+, Platform_Count
- Normalization: StandardScaler
- Visualization: PCA (2 components)

### How to Run
```bash
pip install -r requirements.txt
python app.py
# Open: http://localhost:5000
```

### For Actual Kaggle Dataset
1. Download CSV from Kaggle link above
2. Replace `movies.csv` with the downloaded file
3. The column names must match: Title, Year, Age, Rotten Tomatoes, Netflix, Hulu, Prime Video, Disney+
