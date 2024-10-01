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

def process_obj_files_in_folder(folder_path, csv_df):
    """Processes all .obj files located in subfolders based on class names and calculates their barycenters."""
    barycenters = {}

    for _, row in csv_df.iterrows():
        class_folder = row['class']
        file_name = row['file_name']

        file_path = os.path.join(folder_path, class_folder, file_name)
        if os.path.exists(file_path):  
            vertices = extract_vertices_from_obj(file_path)
            barycenters[file_name] = calculate_barycenter(vertices)
        else:
            print(f"Warning: {file_path} does not exist.")
    
    return barycenters

def update_existing_csv_with_barycenters(csv_file_path, barycenters):
    """Updates the existing CSV file with barycenters for matching files."""

    df = pd.read_csv(csv_file_path)

    for col in ['Barycenter_X', 'Barycenter_Y', 'Barycenter_Z']:
        if col not in df.columns:
            df[col] = None

    df['file_name'] = df['file_name'].str.strip()

    for file_name, (cx, cy, cz) in barycenters.items():
        cleaned_file_name = file_name.strip()

        if cleaned_file_name in df['file_name'].values:
            df.loc[df['file_name'] == cleaned_file_name, ['Barycenter_X', 'Barycenter_Y', 'Barycenter_Z']] = [cx, cy, cz]
        else:
            print(f"Warning: {cleaned_file_name} not found in CSV.")

    df.to_csv(csv_file_path, index=False)
    print(f"Barycenters updated and saved to {csv_file_path}")


folder_path = 'refined_dataset' 
csv_file_path = 'refined_dataset_statistics.csv'  

csv_df = pd.read_csv(csv_file_path)

barycenters = process_obj_files_in_folder(folder_path, csv_df)

update_existing_csv_with_barycenters(csv_file_path, barycenters)




