import os
import open3d as o3d
from pathlib import Path
import csv

def clean_mesh(mesh):
    print(f"Cleaning mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces.")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    print(f"Post-cleaning: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces.")
    return mesh

def aggressive_preprocess(mesh, target_face_count=50000, reduction_factor=0.5):
    print(f"Pre-processing large mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces.")
    mesh = clean_mesh(mesh)
    while len(mesh.triangles) > target_face_count * 1.5:
        new_triangle_count = int(len(mesh.triangles) * reduction_factor)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=new_triangle_count)
        print(f"Decimation step: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces.")
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_face_count)
    print(f"Post pre-processing: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces.")
    return mesh

def refine_meshes(input_path, output_path, min_vertex_count=3000, max_vertex_count=50000, 
                  high_vertex_threshold=50000, heavy_decimation_threshold=50000, max_iterations=20):
    try:
        mesh = o3d.io.read_triangle_mesh(input_path)
        vertices_count = len(mesh.vertices)
        faces_count = len(mesh.triangles)
        if faces_count > heavy_decimation_threshold or vertices_count > heavy_decimation_threshold:
            print(f"Mesh at {input_path} has {vertices_count} vertices and {faces_count} faces, performing aggressive pre-processing...")
            mesh = aggressive_preprocess(mesh, target_face_count=max_vertex_count)
            vertices_count = len(mesh.vertices)
            faces_count = len(mesh.triangles)
        
        iterations = 0
        while (vertices_count < min_vertex_count or vertices_count > max_vertex_count or faces_count > max_vertex_count) and iterations < max_iterations:
            iterations += 1
            if vertices_count < min_vertex_count:
                print(f"Refining mesh {input_path} - Iteration {iterations}")
                mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            elif vertices_count > max_vertex_count or faces_count > max_vertex_count:
                print(f"Simplifying mesh {input_path} - Iteration {iterations}")
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=(max_vertex_count + min_vertex_count) // 2)
            vertices_count = len(mesh.vertices)
            faces_count = len(mesh.triangles)
            print(f"Iteration {iterations}: {vertices_count} vertices, {faces_count} faces")
            if faces_count > heavy_decimation_threshold:
                print(f"Mesh at {input_path} has become too complex during refinement. Performing aggressive simplification...")
                mesh = aggressive_preprocess(mesh, target_face_count=max_vertex_count)
                vertices_count = len(mesh.vertices)
                faces_count = len(mesh.triangles)
                print(f"After aggressive simplification: {vertices_count} vertices, {faces_count} faces")
            if iterations >= max_iterations:
                print(f"Stopping after {max_iterations} iterations. Mesh at {input_path} still has too many faces: {faces_count}")
                break

        if min_vertex_count <= vertices_count <= max_vertex_count and faces_count <= max_vertex_count:
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"Processed {Path(input_path).name}: {vertices_count} vertices, {faces_count} faces")
            return {
                "vertices": vertices_count,
                "faces": faces_count,
                "min_bound": mesh.get_min_bound(),
                "max_bound": mesh.get_max_bound()
            }
        else:
            print(f"Mesh at {input_path} could not be refined to acceptable limits: {vertices_count} vertices, {faces_count} faces")
            return None
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
                    
                    # Extract class name from directory structure
                    class_name = Path(root).stem
                    class_output_dir = os.path.join(output_dir, class_name)
                    Path(class_output_dir).mkdir(parents=True, exist_ok=True)
                    
                    output_path = os.path.join(class_output_dir, file)
                    
                    print(f"Processing {file} in class {class_name}")
                    
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
