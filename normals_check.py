import os
import bpy

path = os.path.abspath(os.path.dirname(__file__))

def process_obj_file(file_path, destination_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=file_path)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent()
    bpy.ops.wm.obj_export(filepath=destination_path, export_materials=False)
    pass

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
