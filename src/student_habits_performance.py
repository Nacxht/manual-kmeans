import os
import pandas as pd

from modules import KMeans

abs_path = os.path.dirname(os.path.abspath(__file__))
csv_path = "datasets/student_habits_performance.csv"
columns = ["social_media_hours", "sleep_hours"]

kmeans = KMeans(
  f"{abs_path}/{csv_path}",
  columns
)

# menampikan 5 baris pertama
df_head = kmeans.get_df_head()
print(f"{df_head}\n")

# menampikan info dataframe
kmeans.get_df_info()
print()

# menampilkan pairplot
# kmeans.pairplot(columns)

# menampilkan descripitive statistics
# seperti mean, median, modus, stdev
for column in columns:
  descriptive_stats = kmeans.get_descriptive_statitics(column)
  print(f"{column}:\n{descriptive_stats}\n")

# data cleaning
# menghapus missing values
kmeans.delete_missing_value()

# menampilkan data yang telah dinormalisasi
x_scaled = kmeans.min_max_normalization()
normalized_df = pd.DataFrame({columns[0]: [x_scaled[0][0]], columns[1]: [x_scaled[0][1]]})

for row in range(len(x_scaled)):
  normalized_df.loc[len(normalized_df)] = [x_scaled[row][0], x_scaled[row][1]]
  
print(normalized_df.head())

# penentuan jumlah cluster (K)
# menggunakan silhouette score
# kmeans.silhouette_visualization(10)

# proses clustering
# dengan jumlah cluster (K) sebanyak 6 cluster
labels, centroid_history, convergence_iter = kmeans.clustering(6)

# log pergerakan centroid
for i in range(len(centroid_history)):
  print(f"Iterasi ke-{i}")
  
  for j in range(len(centroid_history[i])):
    print(f"Cluster ke-{j}: ({centroid_history[i][j][0]:.2f}, {centroid_history[i][j][1]:.2f})")
    pass
  
  print() 

# visualisasi clustering
kmeans.visualize_clustering(labels, centroid_history)