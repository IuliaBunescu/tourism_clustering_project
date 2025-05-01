import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

df = pd.read_csv("../Data/final_cleaned_dataset.csv")

"""
Returns a list of silhouette score for results of k- Means for different k values (from 2 to 10)
Param: @param number_clustes - number of clusters
       @df_features data frame where we apply the k-Means algorithm
"""
def getSilhouetteScoreAndInertiaForKMeans(number_clusters, df_features):
    inertia = []
    silhouette_scores = []
    for n_clusters in number_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=1000)
        cluster_labels = kmeans.fit_predict(df_features)
        silhouette_avg = silhouette_score(df_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertia.append(kmeans.inertia_)
    return silhouette_scores, inertia

"""
Plots the silhouette score for each cluster 
Param: @number_clusters - number of clusters
       @silhouette_values- list of silhouette values
       @xlabel - Title of the x axis from the plot
       @ylabel - Title of the y axis from the plot
       @title - Title of the whole figure
"""
def plotValuesBasedOnClusterNumbers(number_clusters, silhouette_values, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.plot(number_clusters, silhouette_values, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


# Apply t-SNE for 2D projection
def applyTSNE(data, cluster_labels, df):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis', s=30)
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    plt.colorbar(label='Cluster')

    # Add country names as labels
    for i, country in enumerate(df["CountryName"]):
        plt.text(X_tsne[i, 0], X_tsne[i, 1], country, fontsize=8)
    plt.grid(True)
    plt.show()

"""
Computed the Dunn index
"""
def dunn_index(X, labels):
    unique_cluster_labels = np.unique(labels)
    n_clusters = len(unique_cluster_labels)
    # Compute intra-cluster distances (max within each cluster)
    intra_dists = []
    for k in unique_cluster_labels:
        cluster_points = X[labels == k]
        if len(cluster_points) > 1:
            distances = pairwise_distances(cluster_points)
            intra_dists.append(np.max(distances))
        else:
            intra_dists.append(0)
 
    max_intra = np.max(intra_dists)
 
    # Compute inter-cluster distances (min between each pair of clusters)
    inter_dists = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = X[labels == unique_cluster_labels[i]]
            cluster_j = X[labels == unique_cluster_labels[j]]
            distances = pairwise_distances(cluster_i, cluster_j)
            inter_dists.append(np.min(distances))
 
    min_inter = np.min(inter_dists)
 
    return min_inter / max_intra if max_intra != 0 else 0

"""
Compute evaluation measures for a data frame considering the resulted cluster labels
"""
def computeEvaluationMeasures(data, cluster_labels):
    silhouette = silhouette_score(data, cluster_labels)
    dunn = dunn_index(data, cluster_labels)
    davies = davies_bouldin_score(data, cluster_labels)
    calinski = calinski_harabasz_score(data, cluster_labels)
    return silhouette, dunn, davies, calinski
    
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



