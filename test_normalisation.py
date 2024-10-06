import os
import numpy as np
import matplotlib.pyplot as plt

def parse_obj_file(filepath):
    """Parses the vertices from a .obj file."""
    vertices = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Lines starting with 'v ' are vertex definitions
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
    return np.array(vertices)

def compute_bounding_box(vertices):
    """Computes the bounding box of a set of vertices."""
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    bbox_dimensions = max_coords - min_coords  # This gives the size in x, y, z dimensions
    return np.max(bbox_dimensions)

def translate_and_scale_to_unit_cube(vertices):
    """
    Translates the vertices to the origin and scales them to fit within a unit cube.
    
    Parameters:
        vertices (np.ndarray): An array of vertices (x, y, z).
    
    Returns:
        np.ndarray: Normalized vertices (x, y, z).
    """
    # Calculate the barycenter (centroid) of the vertices
    barycenter = np.mean(vertices, axis=0)
    
    # Translate vertices to center at the origin
    translated_vertices = vertices - barycenter
    
    # Compute the bounding box of the translated vertices
    min_coords = np.min(translated_vertices, axis=0)
    max_coords = np.max(translated_vertices, axis=0)
    
    # Compute the scaling factor to fit the shape within a unit cube
    bbox_dimensions = max_coords - min_coords
    scaling_factor = 1.0 / np.max(bbox_dimensions)
    
    # Scale the vertices
    normalized_vertices = translated_vertices * scaling_factor
    
    return normalized_vertices

def process_folder_for_bounding_box(folder_path):
    """Processes all .obj files in the folder and its subfolders, computing max bounding box dimension."""
    max_dimensions = []
    
    # Walk through the folder structure
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.obj'):
                obj_filepath = os.path.join(root, file)
                
                # Parse vertices from the .obj file
                vertices = parse_obj_file(obj_filepath)
                
                if len(vertices) > 0:  # Make sure we have vertices
                    # Normalize the vertices to fit within a unit cube
                    normalized_vertices = translate_and_scale_to_unit_cube(vertices)
                    
                    # Compute the maximum bounding box dimension of the normalized vertices
                    max_bbox_dimension = compute_bounding_box(normalized_vertices)
                    
                    # Append this maximum dimension to the overall list
                    max_dimensions.append(max_bbox_dimension)
    
    return max_dimensions

def plot_bounding_box_histogram(max_dimensions):
    """Plots a histogram of the maximum bounding box dimensions."""
    plt.figure(figsize=(8, 6))
    plt.hist(max_dimensions, bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Maximum Bounding Box Dimensions (Normalized to Unit Cube)')
    plt.xlabel('Bounding Box Dimension')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Main workflow
folder_path = 'normalised_dataset'  # Change this to your folder containing subfolders with .obj files
max_bounding_box_dimensions = process_folder_for_bounding_box(folder_path)
plot_bounding_box_histogram(max_bounding_box_dimensions)
