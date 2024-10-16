import math
import open3d as o3d
import numpy as np
from joblib import Parallel, delayed  

def extract_features(mesh: o3d.geometry.TriangleMesh):
    area = mesh.get_surface_area()
    print(f"Area: {area}")
    volume = get_volume(mesh)
    print(f"Volume: {volume}")
    compactness = get_compactness(area, volume)
    print(f"Compactness: {compactness}")
    rectangularity = volume / mesh.get_oriented_bounding_box().volume()
    print(f"Rectangularity: {rectangularity}")
    convex_hull = mesh.compute_convex_hull()[0]
    convexity = volume / get_volume(convex_hull)
    print(f"Convexity: {convexity}")
    diameter = get_diameter(mesh)
    print(f"Diameter: {diameter}")
    eccentricity = get_eccentricity(mesh)
    print(f"Eccentricity: {eccentricity}")

    #A1, D1 D2 D3 D4
    num_samples = 1000
    bins = 10
    descriptor = extract_shape_descriptors(mesh, num_samples=num_samples, bins=bins)
    print(f"Shape Descriptors (A3, D1, D2, D3, D4): {descriptor}")



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

# diameter (Vectorized)
def get_diameter(mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(mesh.vertices)
    distances = np.linalg.norm(vertices[:, np.newaxis] - vertices, axis=2)
    return np.max(distances)

# eccentricity 
def get_eccentricity(mesh: o3d.geometry.TriangleMesh):
    cov = np.cov(np.asarray(mesh.vertices).T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sorted_eigenvalues = np.abs(np.sort(eigenvalues))
    return sorted_eigenvalues[2] / sorted_eigenvalues[0]

# barycenter (needs recplacement still)
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
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        A3_values.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

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

    # Combine all histograms into a single descriptor
    descriptor = np.concatenate([A3_hist, D1_hist, D2_hist, D3_hist, D4_hist])
    return descriptor


#Load one file
mesh_file = "normalised_v2_dataset/Knife/m726.obj"
mesh = o3d.io.read_triangle_mesh(mesh_file)  
extract_features(mesh)  
