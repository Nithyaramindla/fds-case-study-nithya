import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

print("Loading retail dataset... Shape: (7501, 20)")
# Preprocessing
df_retail = pd.read_csv('retail_transactions.csv', header=None)
binary_matrix = df_retail.notnull().astype(int)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(binary_matrix)
explained_variance = pca.explained_variance_ratio_ * 100
print(f"PCA applied. Explained variance: {sum(explained_variance):.2f}%")

# Models
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_pca)
hierarchical = AgglomerativeClustering(n_clusters=4).fit_predict(X_pca)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_pca)
spectral = SpectralClustering(n_clusters=4, random_state=42, gamma=1.0).fit_predict(X_pca)

print("Clustering execution completed!")
