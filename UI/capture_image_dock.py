from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from UI import file_dialog as fd
import os

class CaptureImage(QtGui.QMainWindow):
    def __init__(self, mainWindow, basedir):
        super(CaptureImage, self).__init__(mainWindow)
        self.main = mainWindow
        self.base_directory = basedir
        frame = pg.QtGui.QFrame()
        layout = pg.QtGui.QGridLayout()
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        frame.setLayout(layout)

        self.setCentralWidget(frame)
        self.setWindowTitle("Capture Image")
        self.setWindowIcon(QtGui.QIcon('images.png'))
        screen = QtGui.QDesktopWidget().availableGeometry()
        self.screen_width = screen.width() // 2
        self.screen_height = screen.height() // 2
        self.setGeometry(100, 100, self.screen_width, self.screen_height)

        self.prepare_captureframes(layout)

        # self.main.viewMenu.addAction(self.toggleViewAction())

        # self.capture_dock = QtGui.QDockWidget("Captured Image", self.main)
        # self.capture_dock.setGeometry(0, 0, 1000, 600)
        # self.capture_dock.setWindowIcon(QtGui.QIcon('./UI/images.png'))
        # self.capture_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        # self.main.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.capture_dock)
        # self.main.viewMenu.addAction(self.capture_dock.toggleViewAction())
        # self.prepare_captureframes(self.capture_dock)
        # self.capture_dock.hide()
        # left, top, right, bottom = 10, 10, 10, 10
        # self.capture_dock.setContentsMargins(left, top, right, bottom)

    def prepare_captureframes(self, capture_dock):
        capture_frame = pg.QtGui.QFrame()
        capture_frame_layout = pg.QtGui.QGridLayout()
        capture_frame.setLayout(capture_frame_layout)

        alreay_captured_frame = pg.QtGui.QFrame()
        alreay_captured_frame_layout = pg.QtGui.QGridLayout()
        alreay_captured_frame.setLayout(alreay_captured_frame_layout)

        capture_dock_frame = pg.QtGui.QFrame()
        capture_dock_frame_layout = pg.QtGui.QGridLayout()
        capture_dock_frame.setLayout(capture_dock_frame_layout)

        capture_dock_frame_layout.addWidget(capture_frame, 0, 0)
        capture_dock_frame_layout.addWidget(alreay_captured_frame, 0, 1)

        capture_dock.addWidget(capture_dock_frame)

        self.l_c = QtGui.QLabel(capture_frame)
        self.l_c.setText("Captured Images Folders")
        self.show_captured_data = QtGui.QPushButton('Select Directory', capture_frame)
        self.show_captured_data.clicked.connect(self.start_reading_from_source_onlick)

        self.l_c_no_image = QtGui.QLabel(capture_frame)
        self.l_c_no_image.setText("Number of Images Captured")
        self.l_c_no_image_no = QtGui.QLabel(capture_frame)
        self.l_c_no_image_no.setText("")

        self.l_a_p = QtGui.QLabel(alreay_captured_frame)
        self.l_a_p.setText("Already Trained Image Folders")

        capture_frame_layout.addWidget(self.l_c, 0, 0)
        capture_frame_layout.addWidget(self.show_captured_data, 0, 1)
        capture_frame_layout.addWidget(self.l_c_no_image, 1, 0)
        capture_frame_layout.addWidget(self.l_c_no_image_no, 1, 1)

        alreay_captured_frame_layout.addWidget(self.l_a_p, 0, 0)

        # self.list_folders()

    def list_folders(self):
        # print(self.main.base_directory)
        dir_list = os.listdir(os.path.join(self.base_directory, "data/captured"))

        for l in dir_list:
            print(l)

    def start_reading_from_source_onlick(self):
        fd_obj = fd.OpenFileDialog()
        fd_obj.selectDirectory()
        print(fd_obj.directory_name)

        if fd_obj.directory_name:
            images_names = os.listdir(fd_obj.directory_name)
            self.l_c_no_image_no.setText(str(len(images_names)))
            # for image_name in images_names:
            #     image_path = os.path.join(fd_obj.directory_name, image_name)
            #     image_file = cv2.imread(image_path)
