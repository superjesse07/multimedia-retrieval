import os
import math
import open3d as o3d
import numpy as np

def get_volume(mesh):
    vertices = np.asarray(mesh.vertices)
    volume = 0
    for (v0, v1, v2) in np.asarray(mesh.triangles):
        v0 = vertices[v0]
        v1 = vertices[v1]
        v2 = vertices[v2]
        volume += signed_volume_triangles(v0, v1, v2)
    return volume

def signed_volume_triangles(v0, v1, v2):
    return np.dot(v0, np.cross(v1, v2)) / 6.0

def get_compactness(area, volume):
    return (area ** 3) / (36 * math.pi * (volume ** 2))

def get_diameter(mesh):
    vertices = np.asarray(mesh.vertices)
    max_dist = 0
    chunk_size = 1000
    for i in range(0, len(vertices), chunk_size):
        distances = np.linalg.norm(vertices[i:i+chunk_size][:, np.newaxis] - vertices, axis=2)
        max_dist = max(max_dist, np.max(distances))
    return max_dist

def get_eccentricity(mesh):
    cov = np.cov(np.asarray(mesh.vertices).T)
    eigenvalues, _ = np.linalg.eig(cov)
    sorted_eigenvalues = np.abs(np.sort(eigenvalues))
    return sorted_eigenvalues[2] / sorted_eigenvalues[0]

def compute_barycenter(vertices):
    return np.mean(vertices, axis=0)

def random_vertex_indices(num_vertices, num_samples, points_per_sample=3):
    return np.random.randint(0, num_vertices, size=(num_samples, points_per_sample))

def compute_A3_D1_D2_D3_D4(vertices, barycenter, random_indices):
    A3_values, D1_values, D2_values, D3_values, D4_values = [], [], [], [], []
    for indices in random_indices:
        v0, v1, v2 = vertices[indices[0]], vertices[indices[1]], vertices[indices[2]]
        vec1 = v1 - v0
        vec2 = v2 - v0
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 > 0 and norm_vec2 > 0:
            cos_angle = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
            A3_values.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        else:
            A3_values.append(0.0)
        D1_values.append(np.linalg.norm(barycenter - v0))
        D2_values.append(np.linalg.norm(v0 - v1))
        D3_values.append(np.sqrt(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))))
        v3 = vertices[np.random.randint(0, len(vertices))]
        volume = np.abs(np.dot(v3 - v0, np.cross(v1 - v0, v2 - v0))) / 6.0
        D4_values.append(np.cbrt(volume))

    return A3_values, D1_values, D2_values, D3_values, D4_values

def compute_histogram(values, bins=10, range_min=0, range_max=1):
    histogram, _ = np.histogram(values, bins=bins, range=(range_min, range_max))
    return histogram / np.sum(histogram)

def extract_shape_descriptors(mesh):
    vertices = np.asarray(mesh.vertices)
    barycenter = compute_barycenter(vertices)
    random_indices_A3_D2_D3_D4 = random_vertex_indices(len(vertices), 100000)
    print("Calling compute_A3_D1_D2_D3_D4...")
    result = compute_A3_D1_D2_D3_D4(vertices, barycenter, random_indices_A3_D2_D3_D4)
    #print("compute_A3_D1_D2_D3_D4 result:", result)
    if len(result) != 5:
        print("Error: compute_A3_D1_D2_D3_D4 did not return the expected 5 values.")
    else:
        A3_values, D1_values, D2_values, D3_values, D4_values = result
    A3_hist = compute_histogram(A3_values, bins=100, range_min=0, range_max=np.pi)
    D2_hist = compute_histogram(D2_values, bins=100, range_min=0, range_max=1)
    D3_hist = compute_histogram(D3_values, bins=100, range_min=0, range_max=1)
    D4_hist = compute_histogram(D4_values, bins=100, range_min=0, range_max=1)
    D1_values = []
    D1_values += compute_D1_samples(vertices, barycenter, sample_size=5000)
    D1_values += compute_D1_face_centroid_samples(mesh, barycenter, sample_size=1500)
    D1_hist = compute_histogram(D1_values, bins=30, range_min=0, range_max=1)
    return A3_hist, D1_hist, D2_hist, D3_hist, D4_hist

def compute_D1_samples(vertices, barycenter, sample_size=5000):
    random_indices = random_vertex_indices(len(vertices), sample_size, points_per_sample=1)
    return [np.linalg.norm(barycenter - vertices[idx[0]]) for idx in random_indices]

def compute_D1_face_centroid_samples(mesh, barycenter, sample_size=1500):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    face_centroids = np.mean(vertices[triangles], axis=1)
    random_indices = np.random.randint(0, len(face_centroids), size=sample_size)
    return [np.linalg.norm(barycenter - face_centroids[idx]) for idx in random_indices]

def process_single_query(file_path):
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            raise ValueError("Mesh has no vertices")
        area = mesh.get_surface_area()
        volume = get_volume(mesh)
        compactness = get_compactness(area, volume)
        rectangularity = volume / mesh.get_oriented_bounding_box().volume()
        convex_hull = mesh.compute_convex_hull()[0]
        convexity = volume / get_volume(convex_hull)
        diameter = get_diameter(mesh)
        eccentricity = get_eccentricity(mesh)
        A3_hist, D1_hist, D2_hist, D3_hist, D4_hist = extract_shape_descriptors(mesh)
        features = {
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
    except Exception as e:
        return None

if __name__ == "__main__":
    query_file_path = "path/to/your/query.obj"  
    query_features = process_single_query(query_file_path)
