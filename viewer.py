import moderngl
from PyQt5 import QtOpenGL
import math
import numpy

from util import load_program
from pathlib import Path
from pyrr import Matrix44
import openmesh


class ModelViewerWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        super(ModelViewerWidget, self).__init__(parent)

        self.bg_color = (0.1, 0.1, 0.1, 0.1)
        self.is_wireframe = False
        self.fov = 60.0
        self.camera_zoom = 2.0

    def initializeGL(self):
        self.ctx = moderngl.create_context()

        self.prog = load_program(Path("assets/default.vert"), self.ctx)
        
        self.set_scene()
        self.set_mesh(openmesh.read_trimesh("dataset/Bird/D00089.obj"))

    def set_scene(self):
        self.light = self.prog['Light']
        self.color = self.prog['Color']
        self.mvp = self.prog['Mvp']
        self.light.value = (1.0, 1.0, 1.0)
        self.color.value = (1.0, 1.0, 1.0, 1.0)

        self.mesh = None

    def paintGL(self):
        self.ctx.clear(*self.bg_color)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.wireframe = self.is_wireframe
        if self.mesh is None:
            return

        self.aspect_ratio = self.width() / max(1.0, self.height())
        proj = Matrix44.perspective_projection(self.fov, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            (0.0, 0.0, self.camera_zoom),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        
        self.mvp.write((proj * lookat).astype('f4'))
        
        self.vao.render()

    def set_mesh(self, new_mesh: openmesh.TriMesh):
        if new_mesh is None:
            self.set_scene()
            return

        self.mesh = new_mesh
        self.mesh.update_normals()

        index_buffer = self.ctx.buffer(numpy.array(self.mesh.face_vertex_indices(), dtype="u4").tobytes())
        vao_content = [(self.ctx.buffer(numpy.array(self.mesh.points(), dtype="f4").tobytes()), '3f', 'in_position'),
                       (self.ctx.buffer(numpy.array(self.mesh.vertex_normals(), dtype="f4").tobytes()), '3f', 'in_normal')]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer, 4)
