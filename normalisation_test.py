import os
import csv
import numpy as np

def load_obj_vertices(filepath):
    vertices = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v '): 
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices).T  


def check_pca_alignment(vertices):
    cov_matrix = np.cov(vertices)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    
    alignment_x = np.abs(np.dot(eigenvectors[:, 0], x_axis))  
    alignment_y = np.abs(np.dot(eigenvectors[:, 1], y_axis))  
    alignment_z = np.abs(np.dot(eigenvectors[:, 2], z_axis))  

    return alignment_x, alignment_y, alignment_z

def process_obj_files(input_dir, output_csv):
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(['File', 'Class', 'Eigenvector 1 (X-axis)', 'Eigenvector 2 (Y-axis)', 'Eigenvector 3 (Z-axis)'])
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.obj'):
                    filepath = os.path.join(root, file)
                    class_name = os.path.basename(os.path.dirname(filepath))

                    vertices = load_obj_vertices(filepath)

                    alignment_x, alignment_y, alignment_z = check_pca_alignment(vertices)

                    writer.writerow([file, class_name, alignment_x, alignment_y, alignment_z])

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    input_directory = 'normalised_v2_dataset' 
    output_csv = 'pca_alignment_results.csv'   
    process_obj_files(input_directory, output_csv)








import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd  

file_path = 'pca_alignment_results.csv'  
pca_results = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))

# Plot histograms for each axis
plt.subplot(3, 1, 1)
sns.histplot(pca_results['Eigenvector 1 (X-axis)'], bins=20, kde=True)
plt.title('Histogram of Eigenvector 1 (X-axis) Alignment')

plt.subplot(3, 1, 2)
sns.histplot(pca_results['Eigenvector 2 (Y-axis)'], bins=20, kde=True)
plt.title('Histogram of Eigenvector 2 (Y-axis) Alignment')

plt.subplot(3, 1, 3)
sns.histplot(pca_results['Eigenvector 3 (Z-axis)'], bins=20, kde=True)
plt.title('Histogram of Eigenvector 3 (Z-axis) Alignment')

plt.tight_layout()
plt.show()

