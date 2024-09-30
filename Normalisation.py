import pandas as pd
import matplotlib.pyplot as plt

# Load the refined_dataset_statistics.csv file
csv_file_path = 'refined_dataset_statistics.csv'  # Update the path as needed
data = pd.read_csv(csv_file_path)

# Create a figure with two subplots for vertices and faces count histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the histogram for vertex count (column 'vertices')
ax1.hist(data['vertices'], bins=30, color='skyblue', edgecolor='black')
ax1.set_title('Vertex Count Distribution')
ax1.set_xlabel('Vertex Count')
ax1.set_ylabel('Frequency')

# Plot the histogram for face count (column 'faces')
ax2.hist(data['faces'], bins=30, color='salmon', edgecolor='black')
ax2.set_title('Face Count Distribution')
ax2.set_xlabel('Face Count')
ax2.set_ylabel('Frequency')

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()
