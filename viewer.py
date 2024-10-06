﻿import os
import numpy as np
from PyQt5 import QtOpenGL, QtCore
import math
from PyQt5.QtGui import QMouseEvent, QWheelEvent
from pathlib import Path
from pyrr import Matrix44, Quaternion
import moderngl
import openmesh
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

from util import load_program, get_barycenter


def grid(size, steps):
    u = np.repeat(np.linspace(-size, size, steps), 2)
    v = np.tile([-size, size], steps)
    w = np.zeros(steps * 2)
    new_grid = np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])

    rotation_matrix = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 0]
    ])
    return np.dot(new_grid, rotation_matrix)


class ModelViewerWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        super(ModelViewerWidget, self).__init__(parent)

        self.setMouseTracking(True)
        self.bg_color = (1.0, 1.0, 1.0, 1.0)  # White background
        self.fov = 60.0
        self.camera_zoom = 2.0
        self.wireframe = False
        self.flat_shading = False 
        self.shaded_mode = True  

        self.object_rotation = Quaternion(Matrix44.identity())
        self.prev_x = 0
        self.prev_y = 0
        self.sensitivity = math.pi / 100

        self.barycenter = [0, 0, 0]
        self.cell = 50
        self.size = 5
        self.grid = grid(self.size, self.cell)
        self.grid_alpha_value = 0.5

        self.wireframe_shortcut = QShortcut(QKeySequence("W"), self)
        self.wireframe_shortcut.activated.connect(self.toggle_wireframe)

        self.shaded_mode_shortcut = QShortcut(QKeySequence("S"), self)
        self.shaded_mode_shortcut.activated.connect(self.toggle_shaded_mode)

        self.flat_shading_shortcut = QShortcut(QKeySequence("F"), self)
        self.flat_shading_shortcut.activated.connect(self.toggle_flat_shading)

    def initializeGL(self):
        self.ctx = moderngl.create_context()

      
        self.prog = load_program(Path("assets/default.vert"), self.ctx)

        self.set_scene()

        self.axis_vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # X-axis (red)
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # Y-axis (green)
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],  # Z-axis (blue)
        ], dtype='f4')

        self.axis_vbo = self.ctx.buffer(self.axis_vertices.tobytes())
        self.axis_vao = self.ctx.simple_vertex_array(self.prog, self.axis_vbo, 'in_position')

    def set_scene(self):
    
        self.light = self.prog['Light']
        self.color = self.prog['Color']
        self.mvp = self.prog['Mvp']
        self.light.value = (1.0, 1.0, 1.0)
        self.color.value = (0.0, 0.0, 0.0, 1.0) 

        self.mesh = None
        self.vbo = self.ctx.buffer(self.grid.astype('f4'))
        self.vao2 = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_position')

    def paintGL(self):
        self.ctx.clear(*self.bg_color)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.aspect_ratio = self.width() / max(1.0, self.height())
        proj = Matrix44.perspective_projection(self.fov, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            (0.0, 0.0, self.camera_zoom),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        self.mvp.write((proj * lookat * self.object_rotation.matrix44).astype('f4'))

        self.color.value = (0.0, 0.0, 0.0, self.grid_alpha_value)
        self.vao2.render(moderngl.LINES)

        self.ctx.line_width = 8.0
        self.color.value = (1.0, 0.0, 0.0, 1.0)
        self.axis_vao.render(moderngl.LINES, vertices=2, first=0)
        self.color.value = (0.0, 1.0, 0.0, 1.0)
        self.axis_vao.render(moderngl.LINES, vertices=2, first=2)
        self.color.value = (0.0, 0.0, 1.0, 1.0)
        self.axis_vao.render(moderngl.LINES, vertices=2, first=4)

        self.ctx.line_width = 1.0

        if self.mesh is None:
            return

        translation = Matrix44.from_translation(self.barycenter)
        self.mvp.write((proj * lookat * self.object_rotation.matrix44 * translation).astype('f4'))

        if self.flat_shading:
            self.mesh.update_face_normals()
            normals = np.repeat(self.mesh.face_normals(), 3, axis=0)
        else:
            self.mesh.update_normals()
            normals = self.mesh.vertex_normals()

        vao_content = [
            (self.ctx.buffer(np.array(self.mesh.points(), dtype="f4").tobytes()), '3f', 'in_position'),
            (self.ctx.buffer(np.array(normals, dtype="f4").tobytes()), '3f', 'in_normal'),
        ]

        index_buffer = self.ctx.buffer(np.array(self.mesh.face_vertex_indices(), dtype="u4").tobytes())
        self.vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer)

        if self.shaded_mode:
            self.color.value = (0.0, 0.0, 0.0, 1.0)
            self.vao.render()

        if self.wireframe:
            self.ctx.wireframe = True
            self.color.value = (1.0, 1.0, 1.0, 1.0)
            self.vao.render()
            self.ctx.wireframe = False






    def set_mesh(self, new_mesh: openmesh.PolyMesh):
        if new_mesh is None:
            self.set_scene()
            return

        self.mesh = new_mesh
        self.mesh.update_normals()
        self.barycenter = -get_barycenter(self.mesh)

        index_buffer = self.ctx.buffer(np.array(self.mesh.face_vertex_indices(), dtype="u4").tobytes())
        vao_content = [(self.ctx.buffer(np.array(self.mesh.points(), dtype="f4").tobytes()), '3f', 'in_position'),
                       (self.ctx.buffer(np.array(self.mesh.vertex_normals(), dtype="f4").tobytes()), '3f', 'in_normal')]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer, 4)

    def resizeGL(self, width, height):
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)

    def zoom(self, event: QWheelEvent):
        self.camera_zoom += -event.angleDelta().y() * 0.01
        self.camera_zoom = max(0.2, self.camera_zoom)
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        self.zoom(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.prev_x = event.x()
            self.prev_y = event.y()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            delta_x = (self.prev_x - event.x()) * self.sensitivity
            delta_y = (self.prev_y - event.y()) * self.sensitivity

            self.object_rotation *= Quaternion.from_x_rotation(delta_y)
            self.object_rotation *= Quaternion.from_y_rotation(delta_x)

            self.update()

            self.prev_x = event.x()
            self.prev_y = event.y()

    def toggle_wireframe(self):
        self.wireframe = not self.wireframe
        print(f"Wireframe mode toggled: {self.wireframe}")
        self.update()

    def toggle_flat_shading(self):
        self.flat_shading = not self.flat_shading
        print(f"Flat shading mode toggled: {self.flat_shading}")
        self.update()

    def toggle_shaded_mode(self):
        self.shaded_mode = not self.shaded_mode
        print(f"Shaded mode toggled: {self.shaded_mode}")
        self.update()


