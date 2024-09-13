import openmesh
from PyQt5 import QtCore, QtWidgets
import sys

from viewer import ModelViewerWidget


class ModelViewerApplication(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        self.openGL = ModelViewerWidget(self)
        self.openGL.setGeometry(0,0,800,600)
        timer = QtCore.QTimer(self)
        timer.setInterval(20)
        timer.timeout.connect(self.openGL.updateGL)
        timer.start()
        
        self.loadButton = QtWidgets.QPushButton("Load model")
        self.loadButton.clicked.connect(self.open_file)
        self.wireframeButton = QtWidgets.QPushButton("Toggle wireframe")
        self.wireframeButton.clicked.connect(self.openGL.toggle_wireframe)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.topBar = QtWidgets.QHBoxLayout()
        self.topBar.addWidget(self.loadButton)
        self.topBar.addWidget(self.wireframeButton)
        self.layout.addLayout(self.topBar)
        self.layout.addWidget(self.openGL)
    
    def open_file(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '', "Mesh files (*.obj *.stl *.ply *.off *.om)")
        if not file_name[0]:
            return
        self.openGL.set_mesh(openmesh.read_trimesh(file_name[0]))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = ModelViewerApplication()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec_())