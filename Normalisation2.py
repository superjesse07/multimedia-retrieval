import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_file_path = 'normalised_dataset_statistics.csv'  
df = pd.read_csv(csv_file_path)

barycenter_columns = ['Barycenter_X', 'Barycenter_Y', 'Barycenter_Z']
df_barycenters = df[barycenter_columns]

plt.figure(figsize=(8, 6))
sns.violinplot(data=df_barycenters)
plt.title("Violin Plot of Barycenter Coordinate Differences")
plt.ylabel("Difference from Origin")
plt.xlabel("Coordinate")
plt.show()






\