import os
import math
import open3d as o3d
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import sys
import vg

# Suppress Open3D warnings 
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Volume 
def get_volume(mesh: o3d.geometry.TriangleMesh,triangles=None):
    vertices = np.asarray(mesh.vertices)
    volume = 0
    if triangles == None:
        triangles = np.asarray(mesh.triangles)
    else:
        triangles = np.asarray(mesh.triangles)[triangles]
    for (v0, v1, v2) in triangles:
        v0 = vertices[v0]
        v1 = vertices[v1]
        v2 = vertices[v2]
        volume += signed_volume_triangles(v0, v1, v2)
    return volume

# Signed volume 
def signed_volume_triangles(v0, v1, v2):
    return np.dot(v0, np.cross(v1, v2)) / 6.0

# Total volume across all parts in the mesh
def compute_total_volume(mesh: o3d.geometry.TriangleMesh):
    labels = np.array(mesh.cluster_connected_triangles()[0]) 
    num_parts = labels.max() + 1 

    total_volume = sum(
        abs(get_volume(mesh,np.where(labels == part_index)))
        for part_index in range(num_parts)
    )
    return total_volume 

# Compactness 
def get_compactness(area: float, volume: float):
    if volume == 0:
        return 0
    return (area ** 3) / (36 * math.pi * (volume ** 2))

# Diameter (Vectorized)
def get_diameter(mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(mesh.vertices)
    max_dist = 0
    chunk_size = 1000
    for i in range(0, len(vertices), chunk_size):
        distances = np.linalg.norm(vertices[i:i+chunk_size][:, np.newaxis] - vertices, axis=2)
        max_dist = max(max_dist, np.max(distances))
    return max_dist

# Eccentricity 
def get_eccentricity(mesh: o3d.geometry.TriangleMesh):
    cov = np.cov(np.asarray(mesh.vertices).T)
    eigenvalues = np.linalg.eigvalsh(cov)  # Use eigvalsh for efficiency with symmetric matrices
    sorted_eigenvalues = np.abs(np.sort(eigenvalues))
    return sorted_eigenvalues[2] / sorted_eigenvalues[0]

# Barycenter
def compute_barycenter(vertices):
    return np.mean(vertices, axis=0)

# Random vertex selection
def random_vertex_indices(num_vertices, num_samples, points_per_sample=4):
    return np.random.randint(0, num_vertices, size=(num_samples, points_per_sample))

# A3, D1, D2, D3, D4 descriptor calculations
def compute_A3_D1_D2_D3_D4(vertices, barycenter, random_indices_all, random_indices_D1, triangle_centers):
    v0 = vertices[random_indices_all[:,0]]
    v1 = vertices[random_indices_all[:,1]]
    v2 = vertices[random_indices_all[:,2]]
    v3 = vertices[random_indices_all[:,3]]
    
    D1_values = np.append(np.linalg.norm(vertices[random_indices_D1].squeeze(),axis=1),np.linalg.norm(triangle_centers,axis=1),axis=0)
    A3_values = vg.angle(v0,v1,units="rad")
    D2_values = np.linalg.norm(v0-v1,axis=1)
    D3_values = np.sqrt(0.5 * np.linalg.norm(np.cross(v1-v0,v2-v0),axis=1))
    D4_values = np.cbrt(np.abs(((v3 - v0) * np.cross(v1-v0, v2-v0)).sum(1)) / 6.0)
    return A3_values, D1_values, D2_values, D3_values, D4_values

def compute_histogram(values, bins=10, range_min=0, range_max=1):
    histogram, _ = np.histogram(values, bins=bins, range=(range_min, range_max))
    return histogram / np.sum(histogram)

def extract_shape_descriptors(mesh, num_samples_all=100000, num_samples_D1=5000, num_triangle_centers=1500, bins_all=50, bins_D1=20):
    vertices = np.asarray(mesh.vertices)
    barycenter = compute_barycenter(vertices)
    random_indices_all = random_vertex_indices(len(vertices), num_samples_all)
    random_indices_D1 = random_vertex_indices(len(vertices), num_samples_D1,points_per_sample=1)
    triangle_centers = vertices[np.asarray(mesh.triangles)].mean(axis=1)
    triangle_center_indices = np.random.choice(len(np.asarray(mesh.triangles)), num_triangle_centers)
    
    A3_values, D1_values, D2_values, D3_values, D4_values = compute_A3_D1_D2_D3_D4(
        vertices, barycenter, random_indices_all, random_indices_D1, triangle_centers[triangle_center_indices]
    )

    A3_hist = compute_histogram(A3_values, bins=bins_all, range_min=0, range_max=np.pi)
    D1_hist = compute_histogram(D1_values, bins=bins_D1, range_min=0, range_max=1)
    D2_hist = compute_histogram(D2_values, bins=bins_all, range_min=0, range_max=1)
    D3_hist = compute_histogram(D3_values, bins=bins_all, range_min=0, range_max=1)
    D4_hist = compute_histogram(D4_values, bins=bins_all, range_min=0, range_max=1)

    return A3_hist, D1_hist, D2_hist, D3_hist, D4_hist

def extract_features(category, file, mesh: o3d.geometry.TriangleMesh):
    area = mesh.get_surface_area()
    volume = compute_total_volume(mesh)
    compactness = get_compactness(area, volume)
    rectangularity = volume / mesh.get_oriented_bounding_box().volume()
    convex_hull = mesh.compute_convex_hull()[0]
    convexity = volume / get_volume(convex_hull)
    diameter = get_diameter(mesh)
    eccentricity = get_eccentricity(mesh)
    bins_all = 50
    bins_D1 = 20

    A3_hist, D1_hist, D2_hist, D3_hist, D4_hist = extract_shape_descriptors(
        mesh, bins_all=bins_all, bins_D1=bins_D1
    )
    
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

def process_file(category, file, file_path):
    # try:
        print(f"Processing file: {file_path}")
        
        with SuppressOutput():
            mesh = o3d.io.read_triangle_mesh(file_path)
            return extract_features(category, file, mesh)
    # except Exception as e:
    #     print(f"Error processing {file_path}: {e}")
    #     return None

def process_directory(base_dir):
    categories = os.listdir(base_dir)
    results = Parallel(n_jobs=multiprocessing.cpu_count() / 2)(
        delayed(process_file)(category, file, os.path.join(base_dir, category, file))
        for category in categories if os.path.isdir(os.path.join(base_dir, category))
        for file in os.listdir(os.path.join(base_dir, category)) if file.endswith('.obj')
    )
    
    output_data = [result for result in results if result is not None]
    output_df = pd.DataFrame(output_data)
    output_df.to_csv("feature_database_final.csv", index=False)


base_dir = "normalised_v2_dataset"
process_directory(base_dir)

#print(process_file("Test","test","normalised_v2_dataset/PlantWildNonTree\m963.obj"))
