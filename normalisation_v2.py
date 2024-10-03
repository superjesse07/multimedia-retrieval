import os
import numpy as np

# Function to load vertices from the .obj file
def load_obj(filepath):
    vertices = []
    obj_lines = []
    with open(filepath, 'r') as file:
        for line in file:
            obj_lines.append(line)
            if line.startswith('v '):  # 'v' defines a vertex
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices).T, obj_lines  # Transpose to get a (3, n_points) matrix, return lines as well

# Function to save the transformed vertices back to .obj format
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

def normalize_shape(vertices, filepath):
    print(f"Processing: {filepath}")
    
    # Step 1: Center the points (Translation Normalization)
    centroid = np.mean(vertices, axis=1, keepdims=True)
    vertices_centered = vertices - centroid

    # Step 2: Compute the covariance matrix and perform PCA
    cov_matrix = np.cov(vertices_centered)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Check if any eigenvalue is close to zero (potential problem)
    eigenvalue_threshold = 1e-6
    if np.any(eigenvalues < eigenvalue_threshold):
        print("Warning: One or more eigenvalues are very small, indicating a flat or degenerate shape.")

    # Align the shape based on PCA (eigenvectors)
    aligned_vertices = np.dot(vertices_centered.T, eigenvectors).T

    # Step 3: Scaling the shape into unit volume (no flipping)
    bounding_box = np.ptp(aligned_vertices, axis=1)
    max_dimension = np.max(bounding_box)
    scale_factor = 1.0 / max_dimension
    scaled_vertices = aligned_vertices * scale_factor

    return scaled_vertices

# Function to process all .obj files and apply normalization
def process_obj_files(input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding directory structure in the output folder
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # Process each .obj file in the current directory
        for file in files:
            if file.endswith('.obj'):
                input_filepath = os.path.join(root, file)
                output_filepath = os.path.join(output_subdir, file)

                # Load vertices and obj lines
                vertices, obj_lines = load_obj(input_filepath)

                # Apply normalization (alignment and scaling, no flipping)
                normalized_vertices = normalize_shape(vertices, input_filepath)

                # Save the normalized .obj file
                save_obj(output_filepath, normalized_vertices, obj_lines)

# Define input and output directories
input_directory = 'normalised_dataset'
output_directory = 'normalised_v2_dataset'

# Process the dataset
process_obj_files(input_directory, output_directory)

print("Normalization process completed.")
