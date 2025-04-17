import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("../Data/final_cleaned_dataset.csv")


######## DBSCAN CLUSTERING
# We identified a small number of countries that significantly deviate from global patterns in the dataset. 
# These outliers are not part of any cluster, indicating that their socio-economic or environmental profiles differ substantially from the rest.


# Extract features & normalize the features & run  with tuned eps and min_samples
features = df.drop(columns=["CountryName", "CountryCode"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

eps_value = 3.5          
min_samples_value = 2    

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
labels = dbscan.fit_predict(scaled_features)

# PCA Visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

plt.figure(figsize=(10, 6))
colors = ['#1f77b4' if label != -1 else '#ff0000' for label in labels]
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=colors, s=80, edgecolors='k')

for i, country in enumerate(df["CountryName"]):
    color = 'red' if labels[i] == -1 else 'black'
    plt.annotate(country, (pca_components[i, 0], pca_components[i, 1]), fontsize=8, color=color)

plt.title("DBSCAN Outlier Detection - PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# T-SNE Visualization
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_components = tsne.fit_transform(scaled_features)

plt.figure(figsize=(10, 6))
plt.scatter(tsne_components[:, 0], tsne_components[:, 1], c=colors, s=80, edgecolors='k')

for i, country in enumerate(df["CountryName"]):
    color = 'red' if labels[i] == -1 else 'black'
    plt.annotate(country, (tsne_components[i, 0], tsne_components[i, 1]), fontsize=8, color=color)

plt.title("DBSCAN Outlier Detection - T-SNE")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# List outlier countries
outlier_countries = df[labels == -1][["CountryName", "CountryCode"]]
print("\n Outlier Countries Detected:")
print(outlier_countries)

# Create cleaned dataset
outlier_df = df[labels != -1].copy()
print(f"\n Cleaned dataset contains {len(outlier_df)} countries (out of {len(df)})")

