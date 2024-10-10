import os
import open3d as o3d
from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt

def clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    return mesh

def aggressive_preprocess(mesh, target_count=5000, reduction_factor=0.5):
    mesh = clean_mesh(mesh)
    while len(mesh.vertices) > target_count * 1.5:
        new_vertex_count = int(len(mesh.vertices) * reduction_factor)
        voxel_size = (new_vertex_count / len(mesh.vertices)) * mesh.get_max_bound().max()
        mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size)
    voxel_size = (target_count / len(mesh.vertices)) * mesh.get_max_bound().max()
    mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size)
    return mesh

def refine_meshes(input_path, output_path, target_count=5000, 
                  heavy_decimation_threshold=50000, max_iterations=20, tolerance=0.2, 
                  min_vertices=100, min_faces=100, max_vertices=10000, max_faces=10000):
    try:
        original_mesh = o3d.io.read_triangle_mesh(input_path)
        mesh = original_mesh
        vertices_count = len(mesh.vertices)
        faces_count = len(mesh.triangles)

        if faces_count > heavy_decimation_threshold or vertices_count > heavy_decimation_threshold:
            mesh = aggressive_preprocess(mesh, target_count=target_count)
            vertices_count = len(mesh.vertices)
            faces_count = len(mesh.triangles)

        iterations = 0
        last_valid_mesh = mesh
        while (abs(vertices_count - target_count) > target_count * tolerance or 
               vertices_count > max_vertices or faces_count > max_faces) and iterations < max_iterations:
            iterations += 1
            if vertices_count < target_count:
                mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            elif vertices_count > target_count:
                current_reduction_factor = target_count / vertices_count
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(len(mesh.triangles) * current_reduction_factor))
            mesh = clean_mesh(mesh)
            vertices_count = len(mesh.vertices)
            faces_count = len(mesh.triangles)

        vertices_count = len(mesh.vertices)
        faces_count = len(mesh.triangles)
        if min_vertices <= vertices_count <= max_vertices and min_faces <= faces_count <= max_faces:
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"Processed {Path(input_path).name}: {vertices_count} vertices, {faces_count} faces")
            return {
                "vertices": vertices_count,
                "faces": faces_count,
                "min_bound": mesh.get_min_bound(),
                "max_bound": mesh.get_max_bound()
            }
        else:
            print(f"Mesh {Path(input_path).name} is outside acceptable range after processing: {vertices_count} vertices, {faces_count} faces")
            # Save original mesh if processed mesh is invalid
            o3d.io.write_triangle_mesh(output_path, original_mesh)
            return {
                "vertices": len(original_mesh.vertices),
                "faces": len(original_mesh.triangles),
                "min_bound": original_mesh.get_min_bound(),
                "max_bound": original_mesh.get_max_bound()
            }
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None

def process_directory(input_dir, output_dir, output_csv):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['file_name', 'class', 'vertices', 'faces', 'min_bound', 'max_bound']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".obj"):
                    input_path = os.path.join(root, file)

                    class_name = Path(root).stem
                    class_output_dir = os.path.join(output_dir, class_name)
                    Path(class_output_dir).mkdir(parents=True, exist_ok=True)

                    output_path = os.path.join(class_output_dir, file)

                    result = refine_meshes(input_path, output_path)

                    if result:
                        writer.writerow({
                            'file_name': file,
                            'class': class_name,
                            'vertices': result['vertices'],
                            'faces': result['faces'],
                            'min_bound': result['min_bound'],
                            'max_bound': result['max_bound']
                        })

input_dir = os.path.join(os.path.dirname(__file__), 'dataset')
output_dir = os.path.join(os.path.dirname(__file__), 'refined_dataset')
output_csv = os.path.join(os.path.dirname(__file__), 'refined_dataset_statistics.csv')

process_directory(input_dir, output_dir, output_csv)
print("Refinement process completed.")

# To display the histogram after refinement
csv_file_path = 'refined_dataset_statistics.csv'  
data = pd.read_csv(csv_file_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(data['vertices'], bins=30, color='skyblue', edgecolor='black')
ax1.set_title('Vertex Count Distribution')
ax1.set_xlabel('Vertex Count')
ax1.set_ylabel('Frequency')

ax2.hist(data['faces'], bins=30, color='salmon', edgecolor='black')
ax2.set_title('Face Count Distribution')
ax2.set_xlabel('Face Count')
ax2.set_ylabel('Frequency')

plt.tight_layout()

plt.show()
