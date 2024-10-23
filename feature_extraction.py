import os
import math
import open3d as o3d
import numpy as np
import pandas as pd
from joblib import Parallel, delayed  

def extract_features(category,file,mesh: o3d.geometry.TriangleMesh):
    area = mesh.get_surface_area()
    volume = get_volume(mesh)
    compactness = get_compactness(area, volume)
    rectangularity = volume / mesh.get_oriented_bounding_box().volume()
    convex_hull = mesh.compute_convex_hull()[0]
    convexity = volume / get_volume(convex_hull)
    diameter = get_diameter(mesh)
    eccentricity = get_eccentricity(mesh)
    num_samples = 1000
    bins = 10
    A3_hist, D1_hist, D2_hist, D3_hist, D4_hist = extract_shape_descriptors(mesh, num_samples=num_samples, bins=bins)
    
    features = {
        "category": category,
        "file": file,
        "area": area,
        "volume": volume,
        "compactness": compactness,
        "rectangularity": rectangularity,
        "convexity": convexity,
        "diameter": diameter,
        "eccentricity": eccentricity,
        "A3": A3_hist,
        "D1": D1_hist,
        "D2": D2_hist,
        "D3": D3_hist,
        "D4": D4_hist,
    }
    
    return features

# Volume
def get_volume(mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(mesh.vertices)
    volume = 0
    for (v0, v1, v2) in np.asarray(mesh.triangles):
        v0 = vertices[v0]
        v1 = vertices[v1]
        v2 = vertices[v2]
        volume += signed_volume_triangles(v0, v1, v2)
    return volume

# Signed volume
def signed_volume_triangles(v0, v1, v2):
    return np.dot(v0, np.cross(v1, v2)) / 6.0

# Compactness 
def get_compactness(area: float, volume: float):
    return (area ** 3) / (36 * math.pi * (volume ** 2))

# Diameter (Vectorized)
def get_diameter(mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(mesh.vertices)
    # Compute pairwise distances in chunks to avoid memory issues
    max_dist = 0
    chunk_size = 1000
    for i in range(0, len(vertices), chunk_size):
        distances = np.linalg.norm(vertices[i:i+chunk_size][:, np.newaxis] - vertices, axis=2)
        max_dist = max(max_dist, np.max(distances))
    return max_dist

# Eccentricity 
def get_eccentricity(mesh: o3d.geometry.TriangleMesh):
    cov = np.cov(np.asarray(mesh.vertices).T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sorted_eigenvalues = np.abs(np.sort(eigenvalues))
    return sorted_eigenvalues[2] / sorted_eigenvalues[0]

# Barycenter
def compute_barycenter(vertices):
    return np.mean(vertices, axis=0)

# Random vertex selection 
def random_vertex_indices(num_vertices, num_samples, points_per_sample=3):
    return np.random.randint(0, num_vertices, size=(num_samples, points_per_sample))

# A3, D1, D2, D3, D4
def compute_A3_D1_D2_D3_D4(vertices, barycenter, random_indices):
    A3_values, D1_values, D2_values, D3_values, D4_values = [], [], [], [], []

    for indices in random_indices:
        v0, v1, v2 = vertices[indices[0]], vertices[indices[1]], vertices[indices[2]]
        
        # A3: Angle between vectors
        vec1 = v1 - v0
        vec2 = v2 - v0
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 > 0 and norm_vec2 > 0:
            cos_angle = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
            A3_values.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        else:
            A3_values.append(0.0)  # Default value when division by zero would occur

        # D1: Distance from barycenter to vertex
        D1_values.append(np.linalg.norm(barycenter - v0))

        # D2: Distance between two random vertices
        D2_values.append(np.linalg.norm(v0 - v1))

        # D3: Area of the triangle formed by three vertices
        D3_values.append(np.sqrt(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))))

        # D4: Cube root of the volume of the tetrahedron formed by four vertices
        v3 = vertices[np.random.randint(0, len(vertices))]
        volume = np.abs(np.dot(v3 - v0, np.cross(v1 - v0, v2 - v0))) / 6.0
        D4_values.append(np.cbrt(volume))

    return A3_values, D1_values, D2_values, D3_values, D4_values

def compute_histogram(values, bins=10, range_min=0, range_max=1):
    histogram, _ = np.histogram(values, bins=bins, range=(range_min, range_max))
    return histogram / np.sum(histogram)

def extract_shape_descriptors(mesh, num_samples=1000, bins=10):
    vertices = np.asarray(mesh.vertices)
    barycenter = compute_barycenter(vertices)

    random_indices = random_vertex_indices(len(vertices), num_samples)
    A3_values, D1_values, D2_values, D3_values, D4_values = compute_A3_D1_D2_D3_D4(vertices, barycenter, random_indices)

    A3_hist = compute_histogram(A3_values, bins=bins, range_min=0, range_max=np.pi)
    D1_hist = compute_histogram(D1_values, bins=bins, range_min=0, range_max=1)
    D2_hist = compute_histogram(D2_values, bins=bins, range_min=0, range_max=1)
    D3_hist = compute_histogram(D3_values, bins=bins, range_min=0, range_max=1)
    D4_hist = compute_histogram(D4_values, bins=bins, range_min=0, range_max=1)


    return  A3_hist, D1_hist, D2_hist, D3_hist, D4_hist

def process_file(category,file,file_path):
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        features = extract_features(category,file,mesh)
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_directory(base_dir):
    categories = os.listdir(base_dir)
    output_data = []
    for category in categories:
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.obj')]
            for file in files:
                file_path = os.path.join(category_path, file)
                print(f"Processing file: {file_path}")
                features = process_file(category,file,file_path)
                if features:
                    output_data.append(features)
    output_df = pd.DataFrame(output_data)
    output_df.to_csv("feature_database.csv", index=False)

# Process the entire dataset
base_dir = "normalised_v2_dataset"
process_directory(base_dir)
