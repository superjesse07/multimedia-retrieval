from pathlib import Path

import numpy as np
# import openmesh
from moderngl import Context


def load_program(file_path: Path, ctx: Context):
    with open(file_path.with_suffix(".vert"), "r") as f:
        vertex_shader = f.read()
    with open(file_path.with_suffix(".frag"), "r") as f:
        fragment_shader = f.read()
    return ctx.program(
        vertex_shader, fragment_shader
    )


# def get_barycenter(mesh: openmesh.PolyMesh):
#     mesh_volume = 0
#     barycenter = [0,0,0]
#     
#     for face in mesh.faces():
#         vertices = np.array([mesh.point(vh) for vh in mesh.fv(face)])
#         center = np.sum(vertices,axis=0) / 4
#         volume = np.dot(vertices[0],np.cross(vertices[1],vertices[2]))
#         mesh_volume += volume
#         barycenter = np.add(barycenter,center * volume)
#     
#     return barycenter / mesh_volume
        