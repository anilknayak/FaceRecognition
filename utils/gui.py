from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import cv2
from UI import menu_frame as mf
from UI import status_frame as sf
from UI import prediction_frame as pf
from UI import setting as sef
from UI import capture_image_dock as cpf
import os
import time

import commuter as comm

class VisualizationWindow(QtGui.QMainWindow):
    def __init__(self):
        super(VisualizationWindow, self).__init__()
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        pg.setConfigOptions(imageAxisOrder='row-major')

        self.commuter = comm.Commuter('GUI')


        self.number_of_faces_found_in_image = 2

        frame = pg.QtGui.QFrame()
        layout = pg.QtGui.QGridLayout()
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        frame.setLayout(layout)

        self.camera_window = pg.GraphicsLayoutWidget(self)

        menu_frame = mf.MenuFrame(self)
        status_frame = sf.StatusFrame(self)
        layout.addWidget(self.camera_window, 0, 0, 1, 1)
        layout.addWidget(self.build_config_frame(), 0, 1, 1, 2)

        self.setCentralWidget(frame)
        self.setWindowTitle("Face Recognition")
        self.setWindowIcon(QtGui.QIcon('./UI/images.png'))
        self.setGeometry(0, 0, 1600, 900)

        self.prediction = pf.Prediction(self)
        self.setting = sef.Setting(self)
        self.capture = cpf.CaptureImage(self)
        self.statusBar().showMessage("Ready")
        # self.rows, self.cols = good_shape(self.current_layer_dimensions[0])

        self.video_capture = None
        self.last_frame = None

        self.set_camera()
        self.build_views()
        self.start_timers()

    def build_config_frame(self):
        self.config_frame = pg.QtGui.QFrame()
        self.config_layout = pg.QtGui.QGridLayout()
        self.config_frame.setLayout(self.config_layout)

        return self.config_frame

    def set_camera(self):
        self.video_capture = cv2.VideoCapture(self.commuter.get_camera())

    def publish_pred_result(self, faces):
        if faces is not None:
            j = 0
            for face in faces:
                frame = self.frames[j]
                j+=1
                face_image = face.image
                images = face.images_pred
                titles = face.labels_pred
                self.publish_single_face_detected(face_image,images,titles,frame)

    def publish_single_face_detected(self,face_image,image,title,frame):
        image_frame = frame[0][0]
        title_frame = frame[0][1]
        title_frame.setText("Captured Face")
        image_frame.setImage(np.squeeze(face_image))

        for i in range(self.commuter.get_no_of_prediction()):
            image_frame = frame[i+1][0]
            title_frame = frame[i+1][1]
            title_frame.setText(title[i])
            image_frame.setImage(image[i])


    def camera_callback(self):
        if self.video_capture:
            # frame = np.array(PIL.ImageGrab.grab())[:, :, ::-1]
            # has_frame = True
            has_frame, frame = self.video_capture.read()

            faces = None
            first = False

            if has_frame:
                if not np.any(self.last_frame):
                    first = True
                # self.last_frame = frame[:, :, ::-1]
                # self.capture_face.face_detection_method = self.recognize.configuration.face_detection_method


                if (self.frame_count % self.frame_interval) == 0:

                    faces = self.commuter.detect(frame, 4)

                    # print("Detected", len(faces))

                    self.number_of_faces_found_in_image = len(faces)
                    # normalized_faces = self.commuter.normalize_face(faces)
                    images = None
                    titles = None

                    prediction_labels = []

                    faces = self.commuter.recognize(faces)

                    # print("Recognized", len(faces))
                    # Check our current fps
                    end_time = time.time()
                    if (end_time - self.start_time) > self.fps_display_interval:
                        self.frame_rate = int(self.frame_count / (end_time - self.start_time))
                        self.start_time = time.time()
                        self.frame_count = 0

                    # if not self.recognize.configuration.number_of_face_identified == len(normalized_faces):
                    #     print('Redesigning the frames')
                    #     self.recognize.configuration.number_of_face_identified = len(normalized_faces)
                    #     self.prediction.prepare_pred_frames(self, self.prediction.prediction_dock)
                    #
                    #
                    # persons = 0
                    # for face_a in normalized_faces:
                    #     pred = {}
                    #     face = face_a[0]
                    #     face_cord = face_a[1]
                    #     images, titles = self.recognize.recognize(face)
                    #     pred['face'] = face
                    #     pred['pred_images'] = images
                    #     pred['pred_labels'] = titles
                    #     persons += 1
                    #     prediction_labels.append(pred)
                    #
                    # if images is not None and titles is not None:
                    #     self.publish_pred_result(images, titles, prediction_labels)
                    # frame = self.capture_face.draw_rectangle(faces, frame)

                # faces has one face but 5 pred labels

                    faces = self.commuter.post_processing(faces)
                    self.publish_pred_result(faces)
                    processed_frame = self.commuter.display(faces, frame)

                    self.camera_image.setImage(processed_frame[:, :, ::-1])
                # if first and self.feature_client:
                # self.do_prediction()

                self.frame_count += 1


    def start_timers(self):
        self.camera_timer = QtCore.QTimer()
        self.start_time = time.time()
        self.frame_count = 0
        self.frame_interval = 3
        self.frame_rate = 0
        self.fps_display_interval = 5
        self.camera_timer.timeout.connect(self.camera_callback)
        self.camera_timer.start(10)

    def build_camera_view(self):

        self.camera_view = self.camera_window.addViewBox()
        self.camera_view.setAspectLocked(True)
        self.camera_view.invertY()
        self.camera_view.invertX()
        self.camera_view.setMouseEnabled(False, False)
        self.camera_image = pg.ImageItem(border='w')
        self.camera_view.addItem(self.camera_image)



    def build_views(self):
        self.build_camera_view()
        # self.build_lastframe_view()
        # self.build_features_view()
        # self.build_detailed_feature_view()

    def select_filter(self, event):
        x = event.pos().x()
        y = event.pos().y()

        row = y // np.ceil(self.current_layer_dimensions[0] * 1.1)
        col = x // np.ceil(self.current_layer_dimensions[1] * 1.1)
        n = self.cols * row + col
        if n < self.current_layer_dimensions[2]:
            print("Selected filter {}".format(n))
            self.selected_filter = int(n)


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])
    z = VisualizationWindow()
    z.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
