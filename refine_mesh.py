import os
import trimesh
import pymeshlab
from pathlib import Path

# Function to refine a mesh using pymeshlab
def refine_mesh(input_path, output_path):
    # Load the mesh
    ms = pymeshlab.MeshSet()
    print('loading files from: ' + input_path)
    ms.load_new_mesh(input_path)
    
    # Check vertex and face count
    vertices_count = ms.current_mesh().vertex_number()
    faces_count = ms.current_mesh().face_number()
    
    if vertices_count < 100 or faces_count < 100:
        # Perform subdivision (refinement)
        ms.meshing_surface_subdivision_midpoint(iterations=2)  # You can adjust iterations as needed
        refined_vertices_count = ms.current_mesh().vertex_number()
        refined_faces_count = ms.current_mesh().face_number()
        
        # Check if the refined mesh is not too large
        if refined_vertices_count <= 50000 and refined_faces_count <= 50000:
            # Save the refined mesh
            ms.save_current_mesh(output_path)
            return refined_vertices_count, refined_faces_count
        else:
            print(f"Mesh at {input_path} is too large after refinement: {refined_vertices_count} vertices, {refined_faces_count} faces")
    else:
        # Save the original mesh if no refinement is needed
        ms.save_current_mesh(output_path)
        return vertices_count, faces_count

import os
import trimesh
import pymeshlab
from pathlib import Path

# Function to refine a mesh using pymeshlab
def refine_mesh(input_path, output_path):
    # Load the mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)
    
    # Check vertex and face count
    vertices_count = ms.current_mesh().vertex_number()
    faces_count = ms.current_mesh().face_number()
    
    if vertices_count < 100 or faces_count < 100:
        # Perform subdivision (refinement)
        ms.meshing_surface_subdivision_midpoint(iterations=2)  # You can adjust iterations as needed
        refined_vertices_count = ms.current_mesh().vertex_number()
        refined_faces_count = ms.current_mesh().face_number()
        
        # Check if the refined mesh is not too large
        if refined_vertices_count <= 50000 and refined_faces_count <= 50000:
            # Save the refined mesh
            ms.save_current_mesh(output_path)
            return refined_vertices_count, refined_faces_count
        else:
            print(f"Mesh at {input_path} is too large after refinement: {refined_vertices_count} vertices, {refined_faces_count} faces")
    else:
        # Save the original mesh if no refinement is needed
        ms.save_current_mesh(output_path)
        return vertices_count, faces_count

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
                
                # Refine the mesh and save the result
                vertices, faces = refine_mesh(input_path, output_path)
                print(f"Processed {relative_path}: {vertices} vertices, {faces} faces")

# Directory containing your dataset
input_dir = 'D:\multimedia-retrieval\dataset'
output_dir = 'D:\multimedia-retrieval\\refined'

# Process the dataset directory
process_directory(input_dir, output_dir)
print("Refinement process completed.")
