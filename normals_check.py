import trimesh
import os

def process_obj_file(file_path, destination_path):
    mesh = trimesh.load(file_path, process=False)
    if isinstance(mesh, trimesh.Trimesh):

        if not mesh.is_watertight:
            mesh.fill_holes()

            # if isinstance(filled_mesh, trimesh.Trimesh):
            #     mesh = filled_mesh

        mesh.fix_normals()

        mesh.export(destination_path)
        print(f"Processed and saved: {destination_path}")
    else:
        print(f"Warning: {file_path} is not a valid Trimesh object and will be skipped.")

def process_directory(source_folder, target_folder):

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".obj"):
                file_path = os.path.join(root, file)

                relative_path = os.path.relpath(file_path, source_folder)
                destination_path = os.path.join(target_folder, relative_path)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)

                print(f"Processing {file_path}...")
                process_obj_file(file_path, destination_path)
                print(f"Finished processing {destination_path}.")

if __name__ == "__main__":
    source_folder = "watertight_dataset"

    target_folder = "hole_normal_dataset"

    process_directory(source_folder, target_folder)
