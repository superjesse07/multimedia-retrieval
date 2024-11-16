import locale
from idlelib.iomenu import encoding

import numpy as np
import open3d as o3
from PyQt5 import QtCore, QtWidgets
import sys
from pathlib import Path
import os
from contextlib import contextmanager

from viewer import ModelViewerWidget
import csv
import refine_mesh
import Normalisation
import normalisation_v2
import fill_holes
#import normals_check
import distance_function


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

        self.layout = QtWidgets.QVBoxLayout(self)
        self.topBar = QtWidgets.QHBoxLayout()
        self.topBar.addWidget(self.loadButton)
        self.topBar.addWidget(self.cleanButton)
        self.topBar.addWidget(self.queryButton)
        self.topBar.addWidget(self.wireframeButton)
        self.layout.addLayout(self.topBar)
        self.layout.addWidget(self.openGL,stretch=1)

        self.query_gl = []
        self.query_results = QtWidgets.QGridLayout()
        self.query_results.setSpacing(0)
        for i in range(0,10):
            vertical = QtWidgets.QVBoxLayout()
            gl = ModelViewerWidget()
            gl.setGeometry(0, 0, 800, 600)
            self.query_results.addLayout(vertical,int(i / 5),i % 5)
            category = QtWidgets.QLabel("None")
            distance = QtWidgets.QLabel("Distance: 0.0")
            vertical.addWidget(category)
            vertical.addWidget(distance)
            vertical.addWidget(gl,stretch=1)
            self.query_gl.append({"gl":gl,"distance":distance,"category":category})
        self.layout.addLayout(self.query_results,stretch=1)
        timer = QtCore.QTimer(self)
        timer.setInterval(20)
        timer.timeout.connect(self.update_gl)
        timer.start()


    
    def update_gl(self):
        self.openGL.updateGL()
        for x in self.query_gl:
            x['gl'].updateGL()

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
        #normals_check.process_obj_file("temp.obj","temp.obj")
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
        for (i,frame) in enumerate(self.query_gl):
            print(f'normalised_v2_dataset/{results.iloc[i]["category"]}/{results.iloc[i]["file"]}')
            frame['gl'].set_mesh(o3.io.read_triangle_mesh(f'normalised_v2_dataset/{results.iloc[i]["category"]}/{results.iloc[i]["file"]}'))
            frame['category'].setText(results.iloc[i]["category"])
            frame['distance'].setText(f"Distance: {results.iloc[i]['combined_distance']}")
            print(results.iloc[i])
        progress.close()

    def open_file(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '', "Mesh files (*.obj *.stl *.ply *.off *.om)")
        if not file_name[0]:
            return
        self.openGL.set_mesh(o3.io.read_triangle_mesh(file_name[0]))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = ModelViewerApplication()
    widget.resize(800, 1000)
    widget.show()
    sys.exit(app.exec_())
