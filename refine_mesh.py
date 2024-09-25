import os
import open3d as o3d
import pandas as pd
from pathlib import Path

# Load the dataset
file_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
data = pd.read_csv(file_path, delimiter=';')

# Function to clean and refine a mesh using Open3D
def clean_and_refine_mesh(input_path, output_path):
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(input_path)
        
        # Check vertex and face count
        vertices_count = len(mesh.vertices)
        faces_count = len(mesh.triangles)
        
        # Ensure the mesh has vertices count below 4000 or above 6000
        if vertices_count < 4000 or vertices_count > 6000:
            print(f"Processing {input_path}...")

            # Perform refinement if mesh is poorly sampled
            if vertices_count < 100 or faces_count < 100:
                # Refine the mesh using midpoint subdivision
                mesh = mesh.subdivide_midpoint(number_of_iterations=2)  # Adjust iterations as needed
                refined_vertices_count = len(mesh.vertices)
                refined_faces_count = len(mesh.triangles)
                
                # Check if the refined mesh is not too large
                if refined_vertices_count <= 50000 and refined_faces_count <= 50000:
                    # Save the refined mesh
                    o3d.io.write_triangle_mesh(output_path, mesh)
                    return refined_vertices_count, refined_faces_count
                else:
                    print(f"Mesh at {input_path} is too large after refinement: {refined_vertices_count} vertices, {refined_faces_count} faces")
                    return vertices_count, faces_count
            else:
                # Save the original mesh if no refinement is needed
                o3d.io.write_triangle_mesh(output_path, mesh)
                return vertices_count, faces_count
        else:
            print(f"Skipping {input_path}: Vertices count is between 4000 and 6000.")
            return None, None
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None, None

# Function to process all .obj files in the given directory and its subdirectories
def process_directory(input_dir, output_dir):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Traverse the directory tree
    for root, dirs, files in os.walk(input_dir):
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
                vertices, faces = clean_and_refine_mesh(input_path, output_path)
                if vertices is not None and faces is not None:
                    print(f"Processed {relative_path}: {vertices} vertices, {faces} faces")
                else:
                    print(f"Skipping {relative_path} due to processing error")

# Directory containing your dataset
input_dir = r'D:\multimedia-retrieval\dataset'
output_dir = r'D:\multimedia-retrieval\refined_dataset'

# Process the dataset directory
process_directory(input_dir, output_dir)
print("Refinement process completed.")
