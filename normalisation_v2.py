import os
import numpy as np

def load_obj(filepath):
    vertices = []
    obj_lines = []
    with open(filepath, 'r') as file:
        for line in file:
            obj_lines.append(line)
            if line.startswith('v '): 
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices).T, obj_lines  

def save_obj(filepath, vertices, original_obj_lines):
    with open(filepath, 'w') as file:
        vertex_idx = 0
        for line in original_obj_lines:
            if line.startswith('v '):
                new_vertex = vertices[:, vertex_idx]
                file.write(f'v {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n')
                vertex_idx += 1
            else:
                file.write(line)

def correct_rotation(eigenvalues,eigenvectors):
    eigenvectors[:,[2,1,0]] = eigenvectors[:,eigenvalues.argsort()]

    for i in range(3):
        if np.dot(eigenvectors[:, i], [1 if i == j else 0 for j in range(3)]) < 0:
            eigenvectors[:, i] *= -1 

    return eigenvectors

def normalize_shape(vertices, filepath):
    print(f"Processing: {filepath}")

    centroid = np.mean(vertices, axis=1, keepdims=True)
    vertices_centered = vertices - centroid
    print(f"Centroid: {centroid.ravel()}")

    cov_matrix = np.cov(vertices_centered)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    print(f"Covariance Matrix:\n{cov_matrix}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}")

    eigenvectors = correct_rotation(eigenvalues, eigenvectors)

    aligned_vertices = np.dot(vertices_centered.T, eigenvectors).T
    print(f"Aligned vertices (first 5 rows):\n{aligned_vertices[:, :5]}")

    bounding_box = np.ptp(aligned_vertices, axis=1)
    print(f"Bounding Box (extent of aligned shape along each axis): {bounding_box}")
    max_dimension = np.max(bounding_box)
    scale_factor = 1.0 / max_dimension
    scaled_vertices = aligned_vertices * scale_factor

    print(f"Scale factor: {scale_factor}")
    print(f"Scaled vertices (first 5 rows):\n{scaled_vertices[:, :5]}")

    return scaled_vertices

def process_obj_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
     
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        
        for file in files:
            if file.endswith('.obj'):
                input_filepath = os.path.join(root, file)
                output_filepath = os.path.join(output_subdir, file)
                vertices, obj_lines = load_obj(input_filepath)

                normalized_vertices = normalize_shape(vertices, input_filepath)

                save_obj(output_filepath, normalized_vertices, obj_lines)

input_directory = 'normalised_dataset'
output_directory = 'normalised_v2_dataset'

process_obj_files(input_directory, output_directory)

print("Normalization process completed.")
