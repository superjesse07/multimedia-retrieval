﻿import pandas as pd
import matplotlib.pyplot as plt

file_path = 'C:\\Users\\Carlijn\\multimedia-retrieval\\dataset.csv'
data = pd.read_csv(file_path, delimiter=';')
print(data.head())

# Average number of faces and vertices
average_faces = data['faces'].mean()
average_vertices = data['vertices'].mean()

print(f'Average number of faces: {average_faces}')
print(f'Average number of vertices: {average_vertices}')

# Histograms for faces and vertices
plt.figure(figsize=(12, 6))

# Histogram for number of faces
plt.subplot(1, 2, 1)
plt.hist(data['faces'], bins=20, edgecolor='black')
plt.title('Number of Faces')
plt.xlabel('Number of Faces')
plt.ylabel('Frequency')

# Histogram for number of vertices
plt.subplot(1, 2, 2)
plt.hist(data['vertices'], bins=20, edgecolor='black')
plt.title('Number of Vertices')
plt.xlabel('Number of Vertices')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('histograms_faces_vertices.png', dpi=300)
plt.show()

# SIgnificant outliers based on standard deviation
threshold_faces = average_faces + 2 * data['faces'].std()
threshold_vertices = average_vertices + 2 * data['vertices'].std()

outliers_faces = data[data['faces'] > threshold_faces]
outliers_vertices = data[data['vertices'] > threshold_vertices]

print(f'Outliers based on faces (more than {threshold_faces}):')
print(outliers_faces)

print(f'Outliers based on vertices (more than {threshold_vertices}):')
print(outliers_vertices)

#shaped should be a few thousand vertices, preferbably around 5000 (4000-6000)
# very small nr vertices (300), go very high first, to 15k-20k then back to 5000, in a loop (multiple time)
# you want more uniform ditribution of points, because if you have a large plain whith no point, you could potentially lose out on vital information regarding the shape