import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import cv2
from UI import result_statistics as rs
import os
class AdvanceSetting(QtGui.QMainWindow):
    def __init__(self, mainWindow, configuration, parent=None):
        super(AdvanceSetting, self).__init__(mainWindow)
        self.main = mainWindow
        self.report_start = 'No'
        self.right_arrow_image = configuration.basedir + "/UI/images/right.png"
        self.configuration = configuration

        frame = pg.QtGui.QFrame()
        layout = pg.QtGui.QGridLayout()
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        frame.setLayout(layout)

        self.setCentralWidget(frame)
        self.setWindowTitle("System Flow Diagram and Setting")
        self.setWindowIcon(QtGui.QIcon('images.png'))

        screen = QtGui.QDesktopWidget().availableGeometry()
        self.screen_width = screen.width() // 2
        self.screen_height = screen.height() // 2
        self.setGeometry(100, 100, self.screen_width, self.screen_height)

        self.prepare_setting_frames(layout)

    def analysis_window(self):
        if self.report_start == 'No':
            self.report_start = 'Yes'
        else:
            self.report_start = 'No'

        self.main.change_config('report', self.report_start)

    def reporting_window(self):
        self.rs_window.show()

    def prepare_setting_frames(self, advsetting):
        config_frame_p_main = pg.QtGui.QFrame()
        config_layout_p_main = pg.QtGui.QVBoxLayout()
        config_frame_p_main.setLayout(config_layout_p_main)
        advsetting.addWidget(config_frame_p_main)


        ##########################################
        # First Horizontal layout
        ##########################################
        first_h_frame = pg.QtGui.QFrame()
        first_h_layout = pg.QtGui.QHBoxLayout()
        first_h_frame.setLayout(first_h_layout)
        config_layout_p_main.addWidget(first_h_frame)

        # p = QtGui.QPalette()
        # gradient = QtGui.QLinearGradient(0, 0, 0, 400)
        # gradient.setColorAt(0.0, QtGui.QColor(240, 240, 240))
        # gradient.setColorAt(1.0, QtGui.QColor(240, 160, 160))
        # p.setBrush(QtGui.QPalette.Window, QtGui.QBrush(gradient))
        # first_h_frame.setPalette(p)

        # ================================================
        # ================================================
        title_camera = u'&Camera'
        camera_group_box = QtGui.QGroupBox(title_camera)
        camera_grouo_box_layout = QtGui.QVBoxLayout()

        self.camera_label = QtGui.QLabel()
        self.camera_label.setText("Select Camera")
        self.camera_select = QtGui.QComboBox()
        cameras = self.detectNumAttachedCvCameras()
        list_of_camera = []
        for i in range(cameras):
            list_of_camera.append(str(i))
        self.camera_select.addItems(list_of_camera)
        self.camera_select.currentIndexChanged.connect(self.selectcamera)

        camera_grouo_box_layout.addWidget(self.camera_label)
        camera_grouo_box_layout.addWidget(self.camera_select)
        camera_group_box.setLayout(camera_grouo_box_layout)
        first_h_layout.addWidget(camera_group_box)
        # ================================================

        fr = pg.QtGui.QFrame()
        lt = pg.QtGui.QGridLayout()
        fr.setLayout(lt)
        label = QtGui.QLabel(fr)
        pixmap = QtGui.QPixmap(self.right_arrow_image)
        label.setPixmap(pixmap)
        lt.addWidget(label)
        first_h_layout.addWidget(fr)

        # ================================================
        title_detector_pre_processing = u'&Detector Pre Processing'
        detector_pre_processing_group_box = QtGui.QGroupBox(title_detector_pre_processing)
        detector_pre_processing_group_box_layout = QtGui.QVBoxLayout()

        self.downsampleing_factor_label = QtGui.QLabel()
        self.downsampleing_factor_label.setText("Down Sampling Factor")
        self.downsampleing_factor = QtGui.QComboBox()
        self.downsampleing_factor.addItems(['1', '2', '3', '4', '5'])
        self.downsampleing_factor.currentIndexChanged.connect(self.downsampleing_factor_change)
        detector_pre_processing_group_box_layout.addWidget(self.downsampleing_factor_label)
        detector_pre_processing_group_box_layout.addWidget(self.downsampleing_factor)



        detector_pre_processing_group_box.setLayout(detector_pre_processing_group_box_layout)
        first_h_layout.addWidget(detector_pre_processing_group_box)

        # ================================================


        fr1 = pg.QtGui.QFrame()
        lt1 = pg.QtGui.QGridLayout()
        fr1.setLayout(lt1)
        label1 = QtGui.QLabel(fr1)
        pixmap1 = QtGui.QPixmap(self.right_arrow_image)
        label1.setPixmap(pixmap1)
        lt1.addWidget(label1)
        first_h_layout.addWidget(fr1)



        # ================================================
        title_detector = u'&Detector'
        detector_group_box = QtGui.QGroupBox(title_detector)
        detector_group_box_layout = QtGui.QVBoxLayout()

        self.face_detection_methods_lb = QtGui.QLabel(config_frame_p_main)
        self.face_detection_methods_lb.setText("Face Detection Method")
        self.face_detection_methods = QtGui.QComboBox()
        self.face_detection_methods.addItems(['mtcnn', 'dlib'])
        index = self.face_detection_methods.findText(self.configuration.detector_type, QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.face_detection_methods.setCurrentIndex(index)
        self.face_detection_methods.currentIndexChanged.connect(self.method_selection)
        detector_group_box_layout.addWidget(self.face_detection_methods_lb)
        detector_group_box_layout.addWidget(self.face_detection_methods)
        detector_group_box.setLayout(detector_group_box_layout)
        first_h_layout.addWidget(detector_group_box)
        # ================================================


        fr2 = pg.QtGui.QFrame()
        lt2 = pg.QtGui.QGridLayout()
        fr2.setLayout(lt2)
        label2 = QtGui.QLabel(fr2)
        pixmap2 = QtGui.QPixmap(self.right_arrow_image)
        label2.setPixmap(pixmap2)
        lt2.addWidget(label2)
        first_h_layout.addWidget(fr2)


        # ================================================
        title_detector_post_processing = u'&Detector Post Processing'
        detector_post_processing_group_box = QtGui.QGroupBox(title_detector_post_processing)
        detector_post_processing_group_box_layout = QtGui.QVBoxLayout()

        self.addpadding_label = QtGui.QLabel()
        self.addpadding_label.setText("Add Padding for Rotation")
        self.addpadding = QtGui.QComboBox()
        self.addpadding.addItems(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90'])
        self.addpadding.currentIndexChanged.connect(self.addpadding_change)
        detector_post_processing_group_box_layout.addWidget(self.addpadding_label)
        detector_post_processing_group_box_layout.addWidget(self.addpadding)


        self.vertical_alignment_checkbox = QtGui.QCheckBox("Rotate Face")
        self.vertical_alignment_checkbox.setChecked(False)
        self.vertical_alignment_checkbox.stateChanged.connect(
            lambda: self.vertical_alignment_checkbox_change(self.vertical_alignment_checkbox))
        detector_post_processing_group_box_layout.addWidget(self.vertical_alignment_checkbox)

        self.increase_box_size_lb = QtGui.QLabel()
        self.increase_box_size_lb.setText("Increase Bounding Box")
        self.sl = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sl.setMinimum(-20)
        self.sl.setMaximum(20)
        self.sl.setValue(0)
        self.sl.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sl.setTickInterval(10)
        self.sl.valueChanged.connect(self.sl_change)

        detector_post_processing_group_box_layout.addWidget(self.increase_box_size_lb)
        detector_post_processing_group_box_layout.addWidget(self.sl)



        detector_post_processing_group_box.setLayout(detector_post_processing_group_box_layout)
        first_h_layout.addWidget(detector_post_processing_group_box)
        # ================================================
        # ================================================

        ##########################################
        # First Horizontal layout
        ##########################################




        ##########################################
        # Second Horizontal layout
        ##########################################
        second_h_frame = pg.QtGui.QFrame()
        second_h_layout = pg.QtGui.QHBoxLayout()
        second_h_frame.setLayout(second_h_layout)
        config_layout_p_main.addWidget(second_h_frame)

        # ================================================
        # ================================================
        title_recognizer_pre_processing = u'&Recognizer Pre Processing'
        recognizer_pre_processing_group_box = QtGui.QGroupBox(title_recognizer_pre_processing)
        recognizer_pre_processing_group_box_layout = QtGui.QVBoxLayout()

        recognizer_pre_processing_group_box.setLayout(recognizer_pre_processing_group_box_layout)
        second_h_layout.addWidget(recognizer_pre_processing_group_box)
        # ================================================


        fr3 = pg.QtGui.QFrame()
        lt3 = pg.QtGui.QGridLayout()
        fr3.setLayout(lt3)
        label3 = QtGui.QLabel(fr3)
        pixmap3 = QtGui.QPixmap(self.right_arrow_image)
        label3.setPixmap(pixmap3)
        lt3.addWidget(label3)
        second_h_layout.addWidget(fr3)
       
        # ================================================
        title_recongnizer = u'&Recognizer'
        recognizer_group_box = QtGui.QGroupBox(title_recongnizer)
        recognizer_group_box_layout = QtGui.QVBoxLayout()

        self.facerecognition_api_lb = QtGui.QLabel(config_frame_p_main)
        self.facerecognition_api_lb.setText("Face Recognition Method")
        self.facerecognition_api = QtGui.QComboBox()
        self.facerecognition_api.addItems(['facenet', 'nn', 'inception_v1', 'inception_v5', 'svm', 'svm_facenet'])
        index1 = self.facerecognition_api.findText(self.configuration.recognizer_type, QtCore.Qt.MatchFixedString)
        if index1 >= 0:
            self.facerecognition_api.setCurrentIndex(index1)
        self.facerecognition_api.currentIndexChanged.connect(self.api_selection)
        recognizer_group_box_layout.addWidget(self.facerecognition_api_lb)
        recognizer_group_box_layout.addWidget(self.facerecognition_api)
        recognizer_group_box.setLayout(recognizer_group_box_layout)
        second_h_layout.addWidget(recognizer_group_box)
        # ================================================


        fr4 = pg.QtGui.QFrame()
        lt4 = pg.QtGui.QGridLayout()
        fr4.setLayout(lt4)
        label4 = QtGui.QLabel(fr4)
        pixmap4 = QtGui.QPixmap(self.right_arrow_image)
        label4.setPixmap(pixmap4)
        lt4.addWidget(label4)
        second_h_layout.addWidget(fr4)
        
        
        # ================================================
        title_recongnizer_post_processing = u'&Recognizer Post Processing'
        recognizer_post_processing_group_box = QtGui.QGroupBox(title_recongnizer_post_processing)
        recognizer_post_processing_group_box_layout = QtGui.QVBoxLayout()

        self.no_of_recog_label = QtGui.QLabel()
        self.no_of_recog_label.setText("Number of Face to Recognize")
        self.no_of_recog = QtGui.QComboBox()
        self.no_of_recog.addItems(['1', '2', '3', '4', 'all'])
        self.no_of_recog.currentIndexChanged.connect(self.selection_recog_change)
        recognizer_post_processing_group_box_layout.addWidget(self.no_of_recog_label)
        recognizer_post_processing_group_box_layout.addWidget(self.no_of_recog)

        self.no_of_pred_label = QtGui.QLabel()
        self.no_of_pred_label.setText("Number of Predictions")
        self.no_of_pred = QtGui.QComboBox()
        self.no_of_pred.addItems(['1', '2', '3', '4', '5'])
        self.no_of_pred.currentIndexChanged.connect(self.number_of_prediction_change)
        recognizer_post_processing_group_box_layout.addWidget(self.no_of_pred_label)
        recognizer_post_processing_group_box_layout.addWidget(self.no_of_pred)

        self.accumulator_checkbox = QtGui.QCheckBox("Accumulate Recognition")
        self.accumulator_checkbox.setChecked(False)
        self.accumulator_checkbox.stateChanged.connect(
            lambda: self.accumulator_checkbox_change(self.accumulator_checkbox))
        recognizer_post_processing_group_box_layout.addWidget(self.accumulator_checkbox)

        self.accumulator_label = QtGui.QLabel()
        self.accumulator_label.setText("No of frames to Accumulate")
        self.accumulator_pred = QtGui.QComboBox()
        self.accumulator_pred.addItems(['2', '10', '15', '20', '30'])
        self.accumulator_pred.currentIndexChanged.connect(self.accumulator_change)
        recognizer_post_processing_group_box_layout.addWidget(self.accumulator_label)
        recognizer_post_processing_group_box_layout.addWidget(self.accumulator_pred)

        self.accumulator_wieghted_checkbox = QtGui.QCheckBox("Weighted Accumulator")
        self.accumulator_wieghted_checkbox.setChecked(False)
        self.accumulator_wieghted_checkbox.stateChanged.connect(
            lambda: self.accumulator_wieghted_checkbox_change(self.accumulator_wieghted_checkbox))
        recognizer_post_processing_group_box_layout.addWidget(self.accumulator_wieghted_checkbox)


        recognizer_post_processing_group_box.setLayout(recognizer_post_processing_group_box_layout)
        second_h_layout.addWidget(recognizer_post_processing_group_box)



        # ================================================
        # ================================================





        ##########################################
        # Second Horizontal layout
        ##########################################


        ##########################################
        # Third Horizontal layout
        ##########################################
        third_h_frame = pg.QtGui.QFrame()
        third_h_layout = pg.QtGui.QHBoxLayout()
        third_h_layout.setSpacing(0)
        third_h_frame.setLayout(third_h_layout)
        config_layout_p_main.addWidget(third_h_frame)

        # ================================================
        # ================================================
        title_display = u'&Display'
        display_group_box = QtGui.QGroupBox(title_display)
        display_group_box_layout = QtGui.QVBoxLayout()

        self.feature_point_checkbox = QtGui.QCheckBox("Show Feature Point")
        self.feature_point_checkbox.setChecked(False)
        self.feature_point_checkbox.stateChanged.connect(
            lambda: self.feature_point_checkbox_change(self.feature_point_checkbox))
        display_group_box_layout.addWidget(self.feature_point_checkbox)





        display_group_box.setLayout(display_group_box_layout)
        third_h_layout.addWidget(display_group_box)
        # ================================================


        fr5 = pg.QtGui.QFrame()
        lt5 = pg.QtGui.QGridLayout()
        fr5.setLayout(lt5)
        label5 = QtGui.QLabel(fr5)
        pixmap5 = QtGui.QPixmap(self.right_arrow_image)
        label5.setPixmap(pixmap5)
        lt5.addWidget(label5)
        third_h_layout.addWidget(fr5)
       
       
        # ================================================
        title_capture_face = u'&Capture Face'
        capture_face_group_box = QtGui.QGroupBox(title_capture_face)
        capture_face_group_box_layout = QtGui.QVBoxLayout()

        self.video_codec_label = QtGui.QLabel(config_frame_p_main)
        self.video_codec_label.setText("Video Codec")
        self.video_codec_combobox = QtGui.QComboBox()
        self.video_codec_combobox.addItems(['XVID','X264'])
        self.video_codec_combobox.currentIndexChanged.connect(self.choose_video_codec)
        capture_face_group_box_layout.addWidget(self.video_codec_label)
        capture_face_group_box_layout.addWidget(self.video_codec_combobox)

        self.capture_face_number_label = QtGui.QLabel(config_frame_p_main)
        self.capture_face_number_label.setText("Capture Faces")
        self.capture_face_number_label_combobox = QtGui.QComboBox()
        self.capture_face_number_label_combobox.addItems(['Near to Camera', 'All Faces'])
        self.capture_face_number_label_combobox.currentIndexChanged.connect(self.capture_face_number_change)
        capture_face_group_box_layout.addWidget(self.capture_face_number_label)
        capture_face_group_box_layout.addWidget(self.capture_face_number_label_combobox)

        capture_face_group_box.setLayout(capture_face_group_box_layout)
        third_h_layout.addWidget(capture_face_group_box)

        ##########################################
        # Third Horizontal layout
        ##########################################

    def accumulator_wieghted_checkbox_change(self, b):
        if b.isChecked() == True:
            self.main.change_config('accumulator_weighted_status', True)
        else:
            self.main.change_config('accumulator_weighted_status', False)

    def accumulator_checkbox_change(self, b):
        if b.isChecked() == True:
            self.main.change_config('accumulator_status', True)
        else:
            self.main.change_config('accumulator_status', False)

    def downsampleing_factor_change(self, i):
        value = self.downsampleing_factor.currentText()
        self.main.change_config('down_sampling_factor', value)

    def accumulator_change(self, i):
        value = self.accumulator_pred.currentText()
        self.main.change_config('accumulator_frame_count', value)

    def sl_change(self):
        size = self.sl.value()
        self.main.change_config('increase_boundingbox', size)
        self.increase_box_size_lb.setText("Increase Bounding Box by " + str(size))

    def addpadding_change(self, i):
        value = self.addpadding.currentText()
        self.main.change_config('add_padding', value)

    def vertical_alignment_checkbox_change(self, b):
        if b.isChecked() == True:
            self.main.change_config('vertical_align_face', True)
        else:
            self.main.change_config('vertical_align_face', False)

    def feature_point_checkbox_change(self, b):
        if b.isChecked() == True:
            self.main.change_config('show_feature_point', True)
        else:
            self.main.change_config('show_feature_point', False)

    def get_result_combo_change(self, i):
        value = self.get_result_combo.currentText()
        self.main.change_config('report', value)

    def button_show_reprt_click(self):
        self.rs_window.show()

    def capture_face_number_change(self):
        self.capture_face_dtl = self.capture_face_number_label_combobox.currentText()
        self.main.change_config('capture_face_dtl', self.capture_face_dtl)

    def choose_video_codec(self):
        self.video_codec = self.video_codec_combobox.currentText()
        self.main.change_config('video_codec', self.video_codec)

    def frame_rate_combobox_change(self):
        print('frame rate selected')
        self.frame_rate = self.frame_rate_combobox.currentText()
        # self.settingwindow.accept_changes('frame_rate', str(self.frame_rate))
        self.main.change_config('frame_rate', str(self.frame_rate))

    def selection_recog_change(self):
        # self.main.change_config('number_of_recognized', int(self.no_of_pred.currentText()))
        # self.main.prediction.prepare_pred_frames(self.main, self.main.prediction.prediction_dock)
        self.main.change_config('number_of_recognized', str(self.no_of_recog.currentText()))

    def api_selection(self, i):
        print('Selecting recognizer')
        # self.settingwindow.accept_changes('recognizer', str(self.facerecognition_api.currentText()))
        self.main.change_config('recognizer', str(self.facerecognition_api.currentText()))

    def method_selection(self, i):
        # self.settingwindow.accept_changes('detector', str(self.face_detection_methods.currentText()))
        self.main.change_config('detector', str(self.face_detection_methods.currentText()))

    def number_of_prediction_change(self, i):
        # self.settingwindow.accept_changes('number_of_prediction', int(self.no_of_pred.currentText()))
        # self.settingwindow.change_prediction_boxes()
        self.main.change_config('number_of_prediction', int(self.no_of_pred.currentText()))
        # self.main.prediction.prepare_pred_frames(self.main, self.main.prediction.prediction_dock)

    def selectcamera(self, i):
        # self.settingwindow.accept_changes('camera', int(self.camera_select.currentText()))
        self.main.change_config('camera', int(self.camera_select.currentText()))

    def detectNumAttachedCvCameras(self):
        number = 0
        while (True):
            try:
                cap = cv2.VideoCapture()
                cap.open(number)
                if not cap.isOpened():
                    break
                else:
                    number += 1
                    cap.release()
                if number > 10:
                    break
            except:
                print('')
        return number