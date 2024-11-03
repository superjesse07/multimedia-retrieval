import os

def parse_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '): 
                parts = line.strip().split()
                face = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                faces.append(face)

    return vertices, faces

def calculate_normal(v1, v2, v3):
    edge1 = [v2[i] - v1[i] for i in range(3)]
    edge2 = [v3[i] - v1[i] for i in range(3)]
    normal = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0]
    ]
    length = sum(n**2 for n in normal)**0.5
    return [n / length for n in normal] if length != 0 else [0, 0, 0]

def orient_normals(vertices, faces):
    normals = [calculate_normal(
        vertices[faces[i][0]], vertices[faces[i][1]], vertices[faces[i][2]]
    ) for i in range(len(faces))]

    aligned_normals = [normals[0]] 
    for i in range(1, len(normals)):
        if sum(aligned_normals[-1][j] * normals[i][j] for j in range(3)) < 0:
            normals[i] = [-n for n in normals[i]]
        aligned_normals.append(normals[i])

    return aligned_normals

def save_oriented_obj(vertices, faces, destination_path):
    with open(destination_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            face_line = f"f {' '.join(str(idx + 1) for idx in face)}"
            file.write(face_line + "\n")

    print(f"Oriented .obj file saved at {destination_path}")

def process_obj_file(file_path, destination_path):
    vertices, faces = parse_obj(file_path)
    aligned_normals = orient_normals(vertices, faces)
    save_oriented_obj(vertices, faces, destination_path)

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
