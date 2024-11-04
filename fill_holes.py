from vedo import *
import numpy as np
import os
from pathlib import Path


def fill_holes(mesh: Mesh):
	mesh = mesh.compute_normals().clean()
	lines = [x.color("red") for x in mesh.boundaries(True,False,False).join_segments()]
	holes = [x.triangulate() for x in mesh.boundaries(True,False,False).join_segments()]
	mesh = merge(mesh,holes).clean().compute_normals()
	return mesh


def process_obj_file(file_path, destination_path):
	original_mesh = Mesh(file_path)
	mesh = fill_holes(original_mesh)
	file_io.save(mesh,destination_path)

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
