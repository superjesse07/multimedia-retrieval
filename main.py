import locale
from idlelib.iomenu import encoding

import numpy as np
import openmesh
from PyQt5 import QtCore, QtWidgets
import sys
from pathlib import Path
from openmesh import PolyMesh
import os
from contextlib import contextmanager

from util import get_barycenter
from viewer import ModelViewerWidget
import csv


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
        timer = QtCore.QTimer(self)
        timer.setInterval(20)
        timer.timeout.connect(self.openGL.updateGL)
        timer.start()

        self.loadButton = QtWidgets.QPushButton("Load model")
        self.loadButton.clicked.connect(self.open_file)
        self.wireframeButton = QtWidgets.QPushButton("Toggle wireframe")
        self.wireframeButton.clicked.connect(self.openGL.toggle_wireframe)
        self.parseButton = QtWidgets.QPushButton("Parse Meshes")
        self.parseButton.clicked.connect(self.parse_mesh_data)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.topBar = QtWidgets.QHBoxLayout()
        self.topBar.addWidget(self.loadButton)
        self.topBar.addWidget(self.wireframeButton)
        self.topBar.addWidget(self.parseButton)
        self.layout.addLayout(self.topBar)
        self.layout.addWidget(self.openGL)

    def open_file(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '', "Mesh files (*.obj *.stl *.ply *.off *.om)")
        if not file_name[0]:
            return
        self.openGL.set_mesh(openmesh.read_polymesh(file_name[0]))

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
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec_())
