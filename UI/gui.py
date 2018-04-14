from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import cv2
from UI import menu_frame as mf
from UI import status_frame as sf
from UI import prediction_frame as pf
from UI import setting as sef
from UI import advance_setting as asef
from UI import advance_setting_2 as asef2
from UI import capture_image_dock as cpf


# import commuter as comm

class VisualizationWindow(QtGui.QMainWindow):
    def __init__(self, basedir, commuter, mainprocess):
        super(VisualizationWindow, self).__init__()
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.base_directory = basedir  # os.path.dirname(os.path.abspath(__file__))
        self.configuration = commuter.context.configuration
        self.commuter = commuter
        self.mainprocess = mainprocess
        self.number_of_faces_found_in_image = 2

        frame = pg.QtGui.QFrame()
        layout = pg.QtGui.QGridLayout()
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        frame.setLayout(layout)

        menu_frame = mf.MenuFrame(self)
        status_frame = sf.StatusFrame(self)

        self.capture_face_flag = False
        self.prediction_flag = False
        self.flow_window_flag = False

        # self.camera_window = pg.GraphicsLayoutWidget()
        # self.camera_view = self.camera_window.addViewBox()
        # self.camera_view.setBackgroundColor((255, 255, 255, 255))
        # self.camera_view.setAspectLocked(True)
        # self.camera_view.invertY()
        # self.camera_view.invertX()
        # self.camera_view.setMouseEnabled(False, False)
        # self.camera_image = pg.ImageItem()  # border='w'
        # self.camera_view.addItem(self.camera_image)
        # layout.addWidget(self.camera_window, 0, 0, 1, 1)

        self.graphics_window = pg.GraphicsView(self)
        self.graphics_window.setBackground(None)
        self.camera_image = pg.ImageItem()  # border='w'
        self.graphics_window.addItem(self.camera_image)
        layout.addWidget(self.graphics_window, 0, 0, 1, 1)


        layout.addWidget(self.build_config_frame(), 0, 1, 1, 2)


        # self.statusBar = QtGui.QStatusBar()
        # self.setStatusBar(self.statusBar)

        self.setCentralWidget(frame)
        self.setWindowTitle("Face Recognition")
        self.setWindowIcon(QtGui.QIcon('images.png'))

        screen = QtGui.QDesktopWidget().availableGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()

        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        self.prediction = pf.Prediction(self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.prediction)
        newActionP = QtGui.QAction(QtGui.QIcon('images.png'), '&Prediction', self)
        newActionP.setShortcut('Ctrl+P')
        newActionP.setStatusTip('Prediction Window')
        newActionP.triggered.connect(self.prediction_window)
        self.viewMenu.addAction(newActionP)


        self.setting = sef.Setting(self, self.configuration)
        self.adv_setting = asef.AdvanceSetting(self, self.configuration)
        self.capture = cpf.CaptureImage(self, self.base_directory)
        self.system_slow = asef2.AdvanceSetting(self, self.configuration)

        # self.rows, self.cols = good_shape(self.current_layer_dimensions[0])
        newAction = QtGui.QAction(QtGui.QIcon('images.png'), '&CaptureFaces', self)
        newAction.setShortcut('Ctrl+N')
        newAction.setStatusTip('Show Capture Faces Details')
        newAction.triggered.connect(self.capture_face)
        self.viewMenu.addAction(newAction)

        system_flow_action = QtGui.QAction(QtGui.QIcon('images.png'), '&System Flow', self)
        system_flow_action.setShortcut('Ctrl+A')
        system_flow_action.setStatusTip('Show System Flow for GUI')
        system_flow_action.triggered.connect(self.system_flow_show)
        self.editMenu.addAction(system_flow_action)

        self.video_capture = None
        self.last_frame = None

        # self.set_camera()
        # self.build_views()
        # self.start_timers()

        self.publish_status("Application is ready to use")

    def prediction_window(self):
        if not self.prediction_flag:
            self.prediction_flag = True
            self.prediction.show()
        else:
            self.prediction_flag = False
            self.prediction.hide()

    def getPredictionNumbers(self):

        return self.configuration.number_of_recognized, self.configuration.number_of_prediction

    def closeEvent(self, event):
        self.mainprocess.stop_process()
        event.accept()
        pg.QtCore.QCoreApplication.instance().quit()

    def capture_face(self):

        if not self.capture_face_flag:
            self.capture_face_flag = True
            self.capture.show()
        else:
            self.capture_face_flag = False
            self.capture.hide()

    def system_flow_show(self):
        if not self.flow_window_flag:
            self.flow_window_flag = True
            self.system_slow.show()
        else:
            self.flow_window_flag = False
            self.system_slow.hide()


    def publish_status(self, message):
        self.statusBar().showMessage(message)

    def build_config_frame(self):
        self.config_frame = pg.QtGui.QFrame()
        self.config_layout = pg.QtGui.QGridLayout()
        self.config_frame.setLayout(self.config_layout)
        return self.config_frame

    def change_config(self, type, value):
        # print('Changing the Action=',type,' and setting Value=',value)
        if type == 'camera':
            self.publish_status("Camera changed from "+ str(self.commuter.context.configuration.camera) + " to " + str(value))
            self.commuter.context.configuration.camera = int(value)
        elif type == 'report':

            self.publish_status("Reporting changed from " + self.commuter.context.configuration.report + " to " + value)
            self.commuter.context.configuration.report = value
        elif type == 'detector':
            self.publish_status(
                "Face Detector Library changed from " + self.commuter.context.configuration.detector_type + " to " + value)
            self.commuter.context.configuration.detector_type = value
        elif type == 'recognizer':
            self.publish_status("Face Recognizer Library changed from " + self.commuter.context.configuration.recognizer_type + " to " + value)
            self.commuter.context.configuration.recognizer_type = value
        elif type == 'number_of_prediction':
            self.publish_status(
                "Number of Prediction changed from " + str(self.commuter.context.configuration.number_of_prediction) + " to " + str(value))
            self.commuter.context.configuration.number_of_prediction = int(value)
            self.prediction.setNumberOfPrediction(self.commuter.context.configuration.number_of_prediction)
            self.prediction.prepare_prediction_frame_onchange()
            self.prediction.change_state()
        elif type == 'action':
            self.publish_status(
                "Action changed from " + self.commuter.context.configuration.action + " to " + value)
            self.commuter.context.configuration.action = value
        elif type == 'number_of_recognized':
            self.publish_status(
                "Number of faces to be recognized changed from " + str(self.commuter.context.configuration.number_of_recognized) + " to " + value)
            self.commuter.context.configuration.number_of_recognized = value
            self.prediction.change_state()
        elif type == 'frame_rate':
            self.publish_status(
                "Frame Rate changed from " + self.commuter.context.configuration.frame_rate + " to " + value)
            self.commuter.context.configuration.frame_rate = value
        elif type == 'video_codec':
            self.publish_status(
                "Video Codec changed from " + self.commuter.context.configuration.video_codec + " to " + value)
            self.commuter.context.configuration.video_codec = value
        elif type == 'capture_face_dtl':
            self.publish_status(
                "Face capture setting changed from " + self.commuter.context.configuration.capture_face_dtl + " to " + value)
            self.commuter.context.configuration.capture_face_dtl = value
        elif type == 'show_feature_point':
            self.publish_status(
                "Show Feature Points on Detected Face " + str(self.commuter.context.configuration.show_feature_points) + " to " + str(value))
            self.commuter.context.configuration.show_feature_points = value
        elif type == 'vertical_align_face':
            self.publish_status(
                "Vertical Alignment of Face " + str(self.commuter.context.configuration.vertical_align_face) + " to " + str(value))
            self.commuter.context.configuration.vertical_align_face = value
        elif type == 'add_padding':
            self.publish_status(
                "Add Padding to Face " + str(
                    self.commuter.context.configuration.add_padding) + " to " + str(value))
            self.commuter.context.configuration.add_padding = int(value)
        elif type == 'increase_boundingbox':
            self.publish_status(
                "Increase bounding box size by  " + str(
                    self.commuter.context.configuration.boundingbox_size_incr_by) + " to " + str(value))
            self.commuter.context.configuration.boundingbox_size_incr_by = int(value)

        elif type == 'down_sampling_factor':
            self.publish_status(
                "Down Sampling of Image changed from " + str(
                    self.commuter.context.configuration.down_sampling_factor) + " to " + str(value))
            self.commuter.context.configuration.down_sampling_factor = int(value)
        elif type == 'accumulator_status':
            self.publish_status(
                "Accumulator Status from " + str(self.commuter.context.configuration.accumulator_status) + " to " + str(
                    value))
            self.commuter.context.configuration.accumulator_status = bool(value)
        elif type == 'accumulator_frame_count':
            self.publish_status(
                "Accumulator Frame Count change from " + str(
                    self.commuter.context.configuration.accumulator_frame_count) + " to " + str(value))
            self.commuter.context.configuration.accumulator_frame_count = int(value)

        elif type == 'accumulator_weighted_status':
            self.publish_status(
                "Accumulator Weighted Status from " + str(self.commuter.context.configuration.accumulator_weighted_status) + " to " + str(
                    value))
            self.commuter.context.configuration.accumulator_weighted_status = bool(value)

        elif type == 'stop':
            ''
    def closeEvent(self, event):
        print('close',event.type)
        self.mainprocess.stop_process()
        pg.QtCore.QCoreApplication.instance().quit()


    def start_capturing_source(self):
        self.configuration.start_camera = True
        self.commuter.initialize_camera(self.source, self.video_file)
        self.mainprocess.run()

    def stop_capturing_source(self):
        self.commuter.reset_accumulator()
        self.mainprocess.stop_process()

    def stop(self):
        self.setting.toggle_start_button()

    def output(self, frame, faces):
        self.camera_image.setImage(frame[:, :, ::-1])

        if faces is not None and len(faces) > 0:
            # print("Publish")
            self.prediction.setNumberOfFaces(len(faces))
            self.prediction.prepare_prediction_frame_onchange()
            self.prediction.display_predicted_faces(faces)
        else:
            # print("Not Publish")
            self.prediction.clear_all_frames()