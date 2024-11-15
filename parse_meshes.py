from pathlib import Path

import numpy as np
import openmesh


def get_barycenter(mesh: openmesh.PolyMesh):
   mesh_volume = 0
   barycenter = [0,0,0]
   
   for face in mesh.faces():
       vertices = np.array([mesh.point(vh) for vh in mesh.fv(face)])
       center = np.sum(vertices,axis=0) / 4
       volume = np.dot(vertices[0],np.cross(vertices[1],vertices[2]))
       mesh_volume += volume
       barycenter = np.add(barycenter,center * volume)
   
   return barycenter / mesh_volume

@contextmanager
def suppress_stderr():
    stderr_fileno = sys.stderr.fileno()
    with os.fdopen(os.dup(stderr_fileno), 'w') as old_stderr:
        sys.stderr.close()
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), stderr_fileno)
        try:
            yield
        finally:
            sys.stderr = old_stderr



# This file only works with python version 3.9 as OpenMesh doesn't have any binaries available for later versions.

# Open mesh prints a lot of noise to stderr so this will keep the console clean
with suppress_stderr():
     meshes = [f for f in Path('dataset').rglob("*.obj")]
     header = ['name', 'class', 'faces', 'vertices', 'face type', 'min x', 'min y', 'min z', 'max x', 'max y', 'max z']
     data = []
     for idx, mesh_path in enumerate(meshes):
         print(f'{idx + 1}/{len(meshes)} -- {mesh_path}')
         mesh_class = mesh_path.parent.name
         mesh: PolyMesh = openmesh.read_polymesh(str(mesh_path))
         is_tri = False
         is_quad = False
         for face in mesh.faces():
             valence = mesh.valence(face)
             if valence == 3:
                 is_tri = True
             else:
                 is_quad = True
 
         vertices = np.array([mesh.point(vh) for vh in mesh.vertices()])
 
         min_point = np.min(vertices, axis=0)
         max_point = np.max(vertices, axis=0)
 
         if is_tri:
             mesh_type = "Tri"
             if is_quad:
                 mesh_type = "Mixed"
         else:
             mesh_type = "Quad"
             
         #print(get_barycenter(mesh))
 
         data.append([
             mesh_path.name,
             mesh_class,
             mesh.n_faces(),
             mesh.n_vertices(),
             mesh_type,
             str(min_point[0]).replace('.',','),
             str(min_point[1]).replace('.',','),
             str(min_point[2]).replace('.',','),
             str(max_point[0]).replace('.',','),
             str(max_point[1]).replace('.',','),
             str(max_point[2]).replace('.',',')
         ])
 
     with open('dataset.csv', 'w', encoding='UTF8', newline='') as f:
         writer = csv.writer(f, delimiter=";")
         writer.writerow(header)
         writer.writerows(data)