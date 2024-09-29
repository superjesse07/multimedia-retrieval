import os
import open3d as o3d
from pathlib import Path
import csv

# Function to refine the meshes using Open3D
def refine_meshes(input_path, output_path, min_vertex_count=2000, max_vertex_count=50000, target_vertex_count=4000, max_iterations=5):
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(input_path)
        
        # Check vertex and face count
        vertices_count = len(mesh.vertices)
        faces_count = len(mesh.triangles)
        
        iterations = 0
        while (vertices_count < min_vertex_count or vertices_count > max_vertex_count or 
               faces_count < min_vertex_count or faces_count > max_vertex_count) and iterations < max_iterations:
            iterations += 1
            if vertices_count < min_vertex_count or faces_count < min_vertex_count:
                # Refine the mesh using midpoint subdivision
                mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            elif vertices_count > max_vertex_count or faces_count > max_vertex_count:
                # Simplify the mesh using quadric decimation
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_vertex_count)
            
            vertices_count = len(mesh.vertices)
            faces_count = len(mesh.triangles)
            print(f"Iteration {iterations}: {vertices_count} vertices, {faces_count} faces")

        # Check if the final mesh is within acceptable limits
        if min_vertex_count <= vertices_count <= max_vertex_count and min_vertex_count <= faces_count <= max_vertex_count:
            # Save the refined mesh
            o3d.io.write_triangle_mesh(output_path, mesh)
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

# Function to process all .obj files in the given directory and its subdirectories
def process_directory(input_dir, output_dir, output_csv):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Traverse the directory tree
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith('.obj'):
                input_path = os.path.join(root, file_name)
                
                # Determine relative path to maintain directory structure in output
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                # Ensure the output subdirectory exists
                output_subdir = os.path.dirname(output_path)
                Path(output_subdir).mkdir(parents=True, exist_ok=True)
                
                # Clean and refine the mesh and save the result
                result = refine_meshes(input_path, output_path)
                if result:
                    result["name"] = file_name
                    result["class"] = os.path.basename(os.path.dirname(input_path))
                    results.append(result)
                    print(f"Processed {relative_path}: {result['vertices']} vertices, {result['faces']} faces")
                else:
                    print(f"Skipping {relative_path} due to processing error or out of range vertices/faces")
    
    # Write results to CSV
    header = ['name', 'class', 'faces', 'vertices', 'min_x', 'min_y', 'min_z', 'max_x', 'max_y', 'max_z']
    with open(output_csv, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        for result in results:
            writer.writerow([
                result['name'],
                result['class'],
                result['faces'],
                result['vertices'],
                str(result['min_bound'][0]).replace('.',','),
                str(result['min_bound'][1]).replace('.',','),
                str(result['min_bound'][2]).replace('.',','),
                str(result['max_bound'][0]).replace('.',','),
                str(result['max_bound'][1]).replace('.',','),
                str(result['max_bound'][2]).replace('.',',')
            ])
    
    print(f"Results have been written to {output_csv}")

# Directory containing your dataset
input_dir = os.path.join(os.path.dirname(__file__), 'dataset')
output_dir = os.path.join(os.path.dirname(__file__), 'refined_dataset')
output_csv = os.path.join(os.path.dirname(__file__), 'refined_dataset_statistics.csv')

# Process the dataset directory
process_directory(input_dir, output_dir, output_csv)
print("Refinement process completed.")
