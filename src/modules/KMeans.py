import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from .Statistics import DescriptiveStatistics
from pandas import DataFrame
from collections import Counter

class KMeans(DescriptiveStatistics):
  def __init__(self, csv_path: str, columns: list[str] | None) -> None:
    super().__init__(csv_path)

    self.k: int = 0
    self.df: 'DataFrame' = self.df[columns] if columns else self.df

  def delete_missing_value(self) -> None:
    self.df = self.df.dropna()

  def min_max_normalization(self):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(self.df)

    return scaled

  def z_score_normalization(self):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(self.df)

    return scaled
  
  def silhouette_visualization(self, k_max_range: int = 10):
    x_scaled = self.min_max_normalization()
    silhouette_scores = []
    k_range = range(2, k_max_range + 1)

    for k in k_range:
      labels, _, _  = self.clustering(k)
      score = silhouette_score(x_scaled, labels)
      silhouette_scores.append(score)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    plt.title("Silhouette Score vs Jumlah Cluster")
    plt.xlabel("Jumlah Cluster")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

  def clustering(self, k: int, max_iter: int = 10):  
    X_scaled = self.min_max_normalization()

    # inisialisasi centroid acak
    np.random.seed(42)
    random_index = np.random.choice(len(X_scaled), k, replace=False)
    centroids = X_scaled[random_index, :]

    centroid_history = [centroids.copy()]
    convergence_iter = 0

    # iterasi K-Means
    for i in range(max_iter):
      # hitung jarak ke centroid
      distances = np.linalg.norm(X_scaled[:, np.newaxis] - centroids, axis=2)
      labels = np.argmin(distances, axis=1)

      # update centroid
      new_centroids = np.array([X_scaled[labels == j].mean(axis=0) for j in range(k)])
      centroid_history.append(new_centroids.copy())

      # konvergensi
      if np.allclose(centroids, new_centroids):
        convergence_iter = i
        break

      centroids = new_centroids
    
    self.k = k
    return labels, centroid_history, convergence_iter

  def visualize_clustering(self, labels, centroid_history) -> bool:
    X_scaled = self.min_max_normalization()
    
    if X_scaled.shape[1] < 2:
      return False
    
    X0, X1 = X_scaled[:, 0], X_scaled[:, 1]
    colors = ["r", "g", "b", "c", "m", "y"]

    plt.figure(figsize=(8, 6))
    plt.scatter(X0, X1, c=labels, cmap='viridis', s=30, alpha=0.6, label='Data')

    for k in range(self.k):
      trajectory = np.array([c[k] for c in centroid_history])

      plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        marker='x',
        color=colors[k % len(colors)],
        label=f'Centroid {k}'
      )

      plt.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        color=colors[k % len(colors)],
        edgecolors='black',
        s=200
      )
    
    plt.title("K-Means Clustering & Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()

    return True