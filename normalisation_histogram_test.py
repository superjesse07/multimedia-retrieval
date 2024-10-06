import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the normalization details CSV file
df = pd.read_csv('normalised_dataset_statistics.csv')

# Calculate the distances from the barycenter to the origin
df['Distance_from_Origin'] = np.sqrt(df['Barycenter_X']**2 + df['Barycenter_Y']**2 + df['Barycenter_Z']**2)

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Distance_from_Origin'], bins=50, range=[-1, 1], color='skyblue', edgecolor='black')
plt.title('Histogram of Distance from Barycenter to Origin (after Normalization)')
plt.xlabel('Distance from Origin')
plt.ylabel('Frequency')
plt.show()
