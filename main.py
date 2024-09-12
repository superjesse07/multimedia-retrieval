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
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.openGL)
        

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = ModelViewerApplication()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec_())