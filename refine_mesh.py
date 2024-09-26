import os
import open3d as o3d
import pandas as pd
from pathlib import Path
import csv
import numpy as np

# Function to refine a mesh using Open3D
def refine_meshes(input_path, output_path):
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(input_path)
        
        # Check vertex and face count
        vertices_count = len(mesh.vertices)
        faces_count = len(mesh.triangles)
        
        # If vertices count is not within the desired range, refine the mesh
        if vertices_count < 4000 or vertices_count > 6000:
            print(f"Processing {input_path}...")

            # Dynamic adjustment based on the current vertices count
            if vertices_count < 300:
                # Go up to a high number of vertices, then downsample
                mesh = mesh.subdivide_midpoint(number_of_iterations=2)
                mesh = mesh.simplify_quadric_decimation(int(len(mesh.vertices) * 2))  # Double the vertices for a start
            elif vertices_count < 4000:
                mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            elif vertices_count > 6000:
                # Downsample for high vertex count
                mesh = mesh.simplify_quadric_decimation(int(len(mesh.vertices) * 0.5))  # Reduce to 50%

            # Final check for vertex count
            refined_vertices_count = len(mesh.vertices)
            refined_faces_count = len(mesh.triangles)

            # Ensure the refined mesh is within limits
            if 4000 <= refined_vertices_count <= 6000:
                # Save the refined mesh
                o3d.io.write_triangle_mesh(output_path, mesh)
                return {
                    "vertices": refined_vertices_count,
                    "faces": refined_faces_count,
                    "min_bound": mesh.get_min_bound(),
                    "max_bound": mesh.get_max_bound()
                }
            else:
                print(f"Refined mesh at {input_path} is out of bounds: {refined_vertices_count} vertices.")
                return {
                    "vertices": vertices_count,
                    "faces": faces_count,
                    "min_bound": mesh.get_min_bound(),
                    "max_bound": mesh.get_max_bound()
                }
        else:
            print(f"Skipping {input_path}: Vertices count is between 4000 and 6000.")
            return None
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None

# Function to process all .obj files in the given directory and its subdirectories
def process_directory(input_dir, output_dir, output_csv):
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
                    print(f"Skipping {relative_path} due to processing error")
    
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
                str(result['min_bound'][0]).replace('.', ','),
                str(result['min_bound'][1]).replace('.', ','),
                str(result['min_bound'][2]).replace('.', ','),
                str(result['max_bound'][0]).replace('.', ','),
                str(result['max_bound'][1]).replace('.', ','),
                str(result['max_bound'][2]).replace('.', ',')
            ])
    
    print(f"Results have been written to {output_csv}")

# Directory containing your dataset
input_dir = os.path.join(os.path.dirname(__file__), 'dataset')
output_dir = os.path.join(os.path.dirname(__file__), 'refined_dataset')
output_csv = os.path.join(os.path.dirname(__file__), 'refined_dataset_statistics.csv')

# Process the dataset directory
process_directory(input_dir, output_dir, output_csv)
print("Refinement process completed.")
