import open3d as o3d
import numpy as np
import os
from pathlib import Path


def fill_holes(mesh: o3d.geometry.TriangleMesh):
	try:
		half_edge_mesh : o3d.geometry.HalfEdgeTriangleMesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
		vertices = np.asarray(mesh.vertices)
		boundaries = half_edge_mesh.get_boundaries()
		centroids = list(map(lambda x: np.mean(vertices[x],axis=0),boundaries))
		triangles = np.asarray(mesh.triangles)
		for (i,c) in enumerate(centroids):
			vertices = np.vstack([vertices,c])
			boundary = np.asarray(boundaries[i])
			new_triangles = np.vstack([boundary,np.roll(boundary,1),np.repeat(len(vertices)-1,len(boundary))]).T
			triangles = np.vstack([triangles,new_triangles])
		
		mesh.vertices = o3d.utility.Vector3dVector(vertices)
		mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
		print(mesh.is_watertight())
	except:
		pass

	return mesh

def process_obj_file(file_path, destination_path):
	original_mesh = o3d.io.read_triangle_mesh(file_path)
	mesh = fill_holes(original_mesh)
	o3d.io.write_triangle_mesh(destination_path, mesh)

def process_directory(source_folder, target_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".obj"):
                file_path = os.path.join(root, file)

                relative_path = os.path.relpath(file_path, source_folder)
                destination_path = os.path.join(target_folder, relative_path)
                Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
                print(f"Processing {file_path}...")
                process_obj_file(file_path, str(Path(destination_path)))
                print(f"Finished processing {destination_path}.")


if __name__ == "__main__":
    source_folder = "refined_dataset"
    target_folder = "watertight_dataset"
    process_directory(source_folder, target_folder)
