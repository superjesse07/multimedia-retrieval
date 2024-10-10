import os
import open3d as o3d
from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt

# To display the histogram after refinement
csv_file_path = 'refined_dataset_statistics.csv'  
data = pd.read_csv(csv_file_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(data['vertices'], bins=30, color='skyblue', edgecolor='black')
ax1.set_title('Vertex Count Distribution')
ax1.set_xlabel('Vertex Count')
ax1.set_ylabel('Frequency')

ax2.hist(data['faces'], bins=30, color='salmon', edgecolor='black')
ax2.set_title('Face Count Distribution')
ax2.set_xlabel('Face Count')
ax2.set_ylabel('Frequency')

plt.tight_layout()

plt.show()
