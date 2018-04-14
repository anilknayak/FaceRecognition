from pyqtgraph.Qt import QtCore, QtGui
from UI import image_widget as img
import numpy as np
class CameraFrame(QtGui.QFrame):
    def __init__(self, mainWindow):
        super(CameraFrame, self).__init__()
        self.main_camera_view = img.ImageWidget(self)
        # self.label = QtGui.QLabel(self)

    def setImage(self, image):
        # h,w,d = np.shape(image)
        # frame = QtGui.QImage(image)
        # pixmap = QtGui.QPixmap(frame)
        # self.label.setPixmap(pixmap)
        self.main_camera_view.setImage(image)



