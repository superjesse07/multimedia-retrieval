import locale
from idlelib.iomenu import encoding

import numpy as np
import open3d as o3
from PyQt5 import QtCore, QtWidgets
import sys
from pathlib import Path
from openmesh import PolyMesh
import os
from contextlib import contextmanager

from util import get_barycenter
from viewer import ModelViewerWidget
import csv
import refine_mesh
import Normalisation
import normalisation_v2
import fill_holes
import normals_check
import distance_function


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


class ModelViewerApplication(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.openGL = ModelViewerWidget(self)
        self.openGL.setGeometry(0, 0, 800, 600)

        self.loadButton = QtWidgets.QPushButton("Load model")
        self.loadButton.clicked.connect(self.open_file)
        self.cleanButton = QtWidgets.QPushButton("Clean Model")
        self.cleanButton.clicked.connect(self.clean_model)
        self.queryButton = QtWidgets.QPushButton("Query model")
        self.queryButton.clicked.connect(self.query_model)
        self.wireframeButton = QtWidgets.QPushButton("Toggle wireframe")
        self.wireframeButton.clicked.connect(self.openGL.toggle_wireframe)
        self.parseButton = QtWidgets.QPushButton("Parse Meshes")
        self.parseButton.clicked.connect(self.parse_mesh_data)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.topBar = QtWidgets.QHBoxLayout()
        self.topBar.addWidget(self.loadButton)
        self.topBar.addWidget(self.cleanButton)
        self.topBar.addWidget(self.queryButton)
        self.topBar.addWidget(self.wireframeButton)
        self.topBar.addWidget(self.parseButton)
        self.layout.addLayout(self.topBar)
        self.layout.addWidget(self.openGL)

        self.query_gl = []
        self.query_results = QtWidgets.QGridLayout()
        for i in range(0,10):
            gl = ModelViewerWidget()
            gl.setGeometry(0, 0, 800, 600)
            self.query_results.addWidget(gl,int(i / 5),i % 5)
            self.query_gl.append(gl)
        self.layout.addLayout(self.query_results)
        timer = QtCore.QTimer(self)
        timer.setInterval(20)
        timer.timeout.connect(self.update_gl)
        timer.start()


    
    def update_gl(self):
        self.openGL.updateGL()
        for x in self.query_gl:
            x.updateGL()

    def clean_model(self):
        if self.openGL.mesh is None:
            return
        progress = QtWidgets.QProgressDialog("Refining Mesh...", "Abort Cleaning",0,5,self)
        progress.setWindowTitle("Cleaning Mesh...")
        o3.io.write_triangle_mesh("temp.obj",self.openGL.mesh)
        progress.show()
        progress.setLabelText("Refining Mesh...")
        refine_mesh.refine_meshes("temp.obj","temp.obj")
        progress.setValue(1)
        progress.setLabelText("Filling Holes...")
        fill_holes.process_obj_file("temp.obj","temp.obj")
        progress.setValue(2)
        progress.setLabelText("Flipping Normals...")
        normals_check.process_obj_file("temp.obj","temp.obj")
        progress.setValue(3)
        progress.setLabelText("Normalizing Mesh...")
        Normalisation.process_obj_file("temp.obj","temp.obj")
        progress.setValue(4)
        normalisation_v2.process_obj_file("temp.obj","temp.obj")
        progress.setValue(5)
        progress.close()
        self.openGL.set_mesh(o3.io.read_triangle_mesh("temp.obj"))

    def query_model(self):
        if self.openGL.mesh is None:
            return
        progress = QtWidgets.QProgressDialog("Querying Mesh...", "",0,1,self)
        progress.setWindowTitle("Querying Mesh...")
        progress.show()
        o3.io.write_triangle_mesh("temp.obj",self.openGL.mesh)
        results = distance_function.query_obj("temp.obj")
        for (i,gl) in enumerate(self.query_gl):
            print(f'normalised_v2_dataset/{results.iloc[i]["category"]}/{results.iloc[i]["file"]}')
            gl.set_mesh(o3.io.read_triangle_mesh(f'normalised_v2_dataset/{results.iloc[i]["category"]}/{results.iloc[i]["file"]}'))
        progress.close()

    def open_file(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '', "Mesh files (*.obj *.stl *.ply *.off *.om)")
        if not file_name[0]:
            return
        self.openGL.set_mesh(o3.io.read_triangle_mesh(file_name[0]))

    def parse_mesh_data(self):
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = ModelViewerApplication()
    widget.resize(800, 1000)
    widget.show()
    sys.exit(app.exec_())
