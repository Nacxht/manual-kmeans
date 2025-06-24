import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import norm, mode
from pandas import DataFrame
from matplotlib import pyplot as plt

class DescriptiveStatistics:
  def __init__(self, csv_path: str) -> None:
    self.df: 'DataFrame' = pd.read_csv(csv_path)
  
  def get_df_head(self) -> 'DataFrame':
    return self.df.head()
  
  def get_df_info(self) -> None:
    self.df.info()
  
  def normality_histogram(self, column_name: str) -> bool:
    if column_name not in self.df.columns:
      return False

    data = np.array(self.df[column_name])
    
    sns.histplot(data, kde=True, stat='density', color='skyblue', label='data')
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(data), np.std(data))

    plt.plot(x, p, 'r', label=f"Distribusi normal dari kolom '{column_name}'")
    plt.title(f"Histogram kolom '{column_name}' dengan kurva normal")
    plt.legend()
    plt.grid(True)
    plt.show()

    return True
  
  def scatter_plot(self, x_column: str, y_column: str) -> bool:
    if x_column not in self.df.columns or y_column not in self.df.columns:
      return False
    
    data_x = np.array(self.df[x_column])
    data_y = np.array(self.df[y_column])

    plt.scatter(data_x, data_y, alpha=0.7)
    plt.title(f"Korelasi kolom '{x_column}' dengan kolom '{y_column}'")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()
  
    return True
  
  def get_descriptive_statitics(self, column_name: str) -> 'DataFrame | bool':
    if column_name not in self.df.columns:
      return False
    
    data = np.array(self.df[column_name])
    data_min, data_max = data.min(), data.max()
    data_mean = np.mean(data)
    data_median = np.median(data)
    data_mode = mode(data)
    data_std = data.std()

    descriptive_data = {
      "min": data_min,
      "max": data_max,
      "mean": data_mean,
      "median": data_median,
      "mode": data_mode,
      "st_dev": data_std
    }

    result_dataframe = pd.DataFrame(descriptive_data)
    return result_dataframe
  
  def pairplot(self, columns: list[str]) -> None:
    data = self.df[columns]

    sns.pairplot(data)
    plt.suptitle(f"Pair Plot Kolom")
    plt.show()
    