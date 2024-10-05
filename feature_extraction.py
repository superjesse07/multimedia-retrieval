import math

import open3d as o3
import numpy as np
import itertools


def extract_features(mesh: o3.geometry.TriangleMesh):
    area = mesh.get_surface_area()
    print(area)
    volume = get_volume(mesh)
    print(volume)
    compactness = get_compactness(area, volume)
    print(compactness)
    rectangularity = volume / mesh.get_oriented_bounding_box().volume()
    print(rectangularity)
    convex_hull = mesh.compute_convex_hull()[0]
    convexity = volume / get_volume(convex_hull)
    print(convexity)
    diameter = get_diameter(mesh)
    print(diameter)
    get_eccentricity(mesh)


def get_volume(mesh: o3.geometry.TriangleMesh):
    vertices = np.asarray(mesh.vertices)
    volume = 0
    for (v0, v1, v2) in np.asarray(mesh.triangles):
        v0 = vertices[v0]
        v1 = vertices[v1]
        v2 = vertices[v2]
        volume += signed_volume_triangles(v0,v1,v2)
    return volume


def signed_volume_triangles(v0, v1, v2):
    return np.dot(v0, np.cross(v1, v2)) / 6.0


def get_compactness(area: float, volume: float):
    return (area ** 3) / (36 * math.pi * (volume ** 2))


def get_diameter(mesh: o3.geometry.TriangleMesh):
    diameter = 0
    vertices = np.asarray(mesh.vertices)
    print(len(vertices))
    for p1 in range(0,len(vertices)):
        print(p1)
        for p2 in range(p1+1,len(vertices)):  
            distance = np.linalg.norm(vertices[p1]-vertices[p2])
            diameter = max(diameter,distance)

    return diameter


def get_eccentricity(mesh: o3.geometry.TriangleMesh):
    cov = np.cov(np.asarray(mesh.vertices))
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    print(eigenvalues)


mesh = o3.io.read_triangle_mesh("normalised_dataset/Bed/D00110.obj")
extract_features(mesh)
