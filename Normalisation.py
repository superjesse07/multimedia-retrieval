import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_vertices_and_faces_from_obj(file_path):
    """Extract vertex coordinates and face definitions from an .obj file."""
    vertices, faces = [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if line.startswith('v '):
                vertices.append(tuple(map(float, parts[1:4])))
            elif line.startswith('f '):
                face = [int(index.split('/')[0]) - 1 for index in parts[1:]]
                faces.append(face)
    return vertices, faces

def calculate_vertex_barycenter(vertices):
    """Calculate the barycenter (centroid) of a set of vertices."""
    return tuple(np.mean(np.array(vertices), axis=0))

def translate_to_origin(vertices, barycenter):
    """Translate vertices so that the barycenter coincides with the origin."""
    bx, by, bz = barycenter
    return [(x - bx, y - by, z - bz) for x, y, z in vertices]

def calculate_bounding_box(vertices):
    """Calculate the bounding box of a set of vertices."""
    vertex_array = np.array(vertices)
    min_vals = vertex_array.min(axis=0)
    max_vals = vertex_array.max(axis=0)
    return (*min_vals, *max_vals)

def calculate_scaling_factor(bounding_box):
    """Calculate the scaling factor to fit the shape within a unit-sized cube."""
    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box
    return 1 / max(max_x - min_x, max_y - min_y, max_z - min_z)

def scale_vertices(vertices, scaling_factor):
    """Scale vertices uniformly by the given scaling factor."""
    return [(x * scaling_factor, y * scaling_factor, z * scaling_factor) for x, y, z in vertices]

def save_vertices_and_faces_to_obj(file_path, vertices, faces):
    """Save vertices and faces to an .obj file."""
    with open(file_path, 'w') as file:
        file.writelines([f"v {x} {y} {z}\n" for x, y, z in vertices])
        file.writelines([f"f {' '.join([str(idx + 1) for idx in face])}\n" for face in faces])

def process_obj_files_in_folder(folder_path, csv_df, output_folder):
    """Process all .obj files based on class names, normalize, and save."""
    normalized_shapes = {}

    for _, row in csv_df.iterrows():
        class_folder, file_name = row['class'], row['file_name']
        input_file_path = os.path.join(folder_path, class_folder, file_name)
        output_class_folder = os.path.join(output_folder, class_folder)
        output_file_path = os.path.join(output_class_folder, file_name)

        if os.path.exists(input_file_path):
            os.makedirs(output_class_folder, exist_ok=True)

            # Process vertices and faces
            vertices, faces = extract_vertices_and_faces_from_obj(input_file_path)
            barycenter = calculate_vertex_barycenter(vertices)
            translated_vertices = translate_to_origin(vertices, barycenter)
            bounding_box = calculate_bounding_box(translated_vertices)
            scaling_factor = calculate_scaling_factor(bounding_box)
            scaled_vertices = scale_vertices(translated_vertices, scaling_factor)
            final_barycenter = calculate_vertex_barycenter(scaled_vertices)

            # Save normalised shape
            save_vertices_and_faces_to_obj(output_file_path, scaled_vertices, faces)
            print(input_file_path)
            normalized_shapes[file_name] = {
                "vertices": scaled_vertices,
                "faces": faces,
                "barycenter": final_barycenter,
                "bounding_box": bounding_box,
                "scaling_factor": scaling_factor
            }
        else:
            print(f"Warning: {input_file_path} does not exist.")

    return normalized_shapes

def process_obj_file(input_path,output_path):
    # Process vertices and faces
    vertices, faces = extract_vertices_and_faces_from_obj(input_path)
    barycenter = calculate_vertex_barycenter(vertices)
    translated_vertices = translate_to_origin(vertices, barycenter)
    bounding_box = calculate_bounding_box(translated_vertices)
    scaling_factor = calculate_scaling_factor(bounding_box)
    scaled_vertices = scale_vertices(translated_vertices, scaling_factor)
    final_barycenter = calculate_vertex_barycenter(scaled_vertices)

    # Save normalised shape
    save_vertices_and_faces_to_obj(output_path, scaled_vertices, faces)

def save_normalization_details_to_csv(normalized_shapes, output_csv_file):
    """Save normalization details to a new CSV file."""
    data = [{
        "file_name": file_name,
        "Barycenter_X": details["barycenter"][0],
        "Barycenter_Y": details["barycenter"][1],
        "Barycenter_Z": details["barycenter"][2],
        "Scaling_Factor": details["scaling_factor"],
        "Num_Vertices": len(details["vertices"]),
        "Num_Faces": len(details["faces"])
    } for file_name, details in normalized_shapes.items()]

    pd.DataFrame(data).to_csv(output_csv_file, index=False)

if __name__ == "__main__":
    # Main
    folder_path = 'hole_normal_dataset'
    csv_file_path = 'refined_dataset_statistics.csv'
    output_folder = 'normalised_dataset'
    output_csv_file = 'normalised_dataset_statistics.csv'

    csv_df = pd.read_csv(csv_file_path)
    normalized_shapes = process_obj_files_in_folder(folder_path, csv_df, output_folder)
    save_normalization_details_to_csv(normalized_shapes, output_csv_file)

    # Plot barycenter coordinate differences
    df = pd.read_csv(output_csv_file)
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df[['Barycenter_X', 'Barycenter_Y', 'Barycenter_Z']])
    plt.title("Barycenter Coordinate Differences")
    plt.ylabel("Difference from Origin")
    plt.xlabel("Coordinate")
    plt.show()
