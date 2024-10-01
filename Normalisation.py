import os
import pandas as pd

def extract_vertices_from_obj(file_path):
    """Extracts vertex coordinates from an .obj file."""
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
    return vertices

def calculate_barycenter(vertices):
    """Calculates the barycenter (centroid) of a set of vertices."""
    n = len(vertices)
    if n == 0:
        return (0, 0, 0)
    cx, cy, cz = map(lambda c: sum(c) / n, zip(*vertices))
    return (cx, cy, cz)

def translate_to_origin(vertices, barycenter):
    """Translates vertices so that the barycenter coincides with the origin."""
    bx, by, bz = barycenter
    translated_vertices = [(x - bx, y - by, z - bz) for x, y, z in vertices]
    # Recalculate barycenter to ensure it's at the origin
    new_barycenter = calculate_barycenter(translated_vertices)
    return translated_vertices, new_barycenter

def calculate_bounding_box(vertices):
    """Calculates the bounding box of a set of vertices."""
    min_x = min(vertices, key=lambda v: v[0])[0]
    max_x = max(vertices, key=lambda v: v[0])[0]
    min_y = min(vertices, key=lambda v: v[1])[1]
    max_y = max(vertices, key=lambda v: v[1])[1]
    min_z = min(vertices, key=lambda v: v[2])[2]
    max_z = max(vertices, key=lambda v: v[2])[2]
    return (min_x, max_x, min_y, max_y, min_z, max_z)

def calculate_scaling_factor(bounding_box):
    """Calculates the scaling factor to fit the shape within a unit-sized cube."""
    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box
    max_extent = max(max_x - min_x, max_y - min_y, max_z - min_z)
    return 1 / max_extent if max_extent != 0 else 1

def scale_vertices(vertices, scaling_factor):
    """Scales vertices uniformly by the given scaling factor."""
    return [(x * scaling_factor, y * scaling_factor, z * scaling_factor) for x, y, z in vertices]

def process_obj_files_in_folder(folder_path, csv_df):
    """Processes all .obj files located in subfolders based on class names and normalizes their shapes."""
    normalized_shapes = {}

    for _, row in csv_df.iterrows():
        class_folder = row['class']
        file_name = row['file_name']

        file_path = os.path.join(folder_path, class_folder, file_name)
        if os.path.exists(file_path):
            print(f"Processing {file_name} in {class_folder}")
            vertices = extract_vertices_from_obj(file_path)
            barycenter = calculate_barycenter(vertices)
            print(f"Original Barycenter: {barycenter}")
            
            translated_vertices, new_barycenter = translate_to_origin(vertices, barycenter)
            print(f"New Barycenter after Translation: {new_barycenter}")
            
            if new_barycenter != (0.0, 0.0, 0.0):
                print(f"Error: Translation failed for {file_name}")
                continue

            bounding_box = calculate_bounding_box(translated_vertices)
            scaling_factor = calculate_scaling_factor(bounding_box)
            scaled_vertices = scale_vertices(translated_vertices, scaling_factor)
            
            new_barycenter = calculate_barycenter(scaled_vertices)
            print(f"Barycenter after Scaling: {new_barycenter}")
            
            if new_barycenter != (0.0, 0.0, 0.0):
                print(f"Error: Barycenter not at origin after scaling for {file_name}")
                continue

            normalized_shapes[file_name] = {
                "vertices": scaled_vertices,
                "barycenter": new_barycenter,
                "bounding_box": bounding_box,
                "scaling_factor": scaling_factor
            }
        else:
            print(f"Warning: {file_path} does not exist.")
    
    return normalized_shapes

def save_normalization_details_to_csv(normalized_shapes, output_csv_file):
    """Saves the normalization details to a new CSV file."""
    data = []
    for file_name, details in normalized_shapes.items():
        barycenter = details["barycenter"]
        scaling_factor = details["scaling_factor"]
        data.append({
            "file_name": file_name,
            "Barycenter_X": barycenter[0],
            "Barycenter_Y": barycenter[1],
            "Barycenter_Z": barycenter[2],
            "Scaling_Factor": scaling_factor
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv_file, index=False)
    print(f"Normalization details saved to {output_csv_file}")

folder_path = 'refined_dataset'
csv_file_path = 'refined_dataset_statistics.csv'
output_csv_file = 'normalized_shapes_details.csv'

csv_df = pd.read_csv(csv_file_path)

normalized_shapes = process_obj_files_in_folder(folder_path, csv_df)

save_normalization_details_to_csv(normalized_shapes, output_csv_file)
