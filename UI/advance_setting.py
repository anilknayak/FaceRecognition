import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import cv2
from UI import result_statistics as rs

class AdvanceSetting:
    def __init__(self, mainWindow, configuration):
        self.main = mainWindow
        self.report_start = 'No'
        # self.settingwindow = settingwindow
        self.configuration = configuration
        self.adv_setting_dock = QtGui.QDockWidget("Advance Setting", self.main)

        screen = QtGui.QDesktopWidget().availableGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height() - 200

        self.adv_setting_dock.setGeometry(100, 100, self.screen_height, self.screen_width)
        self.adv_setting_dock.setWindowIcon(QtGui.QIcon('images.png'))
        self.adv_setting_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        self.main.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.adv_setting_dock)
        self.main.editMenu.addAction(self.adv_setting_dock.toggleViewAction())

        self.rs_window = rs.OpenResultStatistics(configuration.basedir)
        # self.settingwindow.advsetting_doc = self.adv_setting_dock
        left, top, right, bottom = 10, 10, 10, 10
        self.adv_setting_dock.setContentsMargins(left, top, right, bottom)

        self.prepare_setting_frames(self.adv_setting_dock)
        self.adv_setting_dock.hide()

        newAction = QtGui.QAction(QtGui.QIcon('images.png'), '&ShowReport', self.main)
        newAction.setShortcut('Ctrl+R')
        newAction.setStatusTip('Display Report')
        newAction.triggered.connect(self.reporting_window)
        self.main.reporting.addAction(newAction)

        newAction1 = QtGui.QAction(QtGui.QIcon('images.png'), '&RecordAnalysis', self.main)
        newAction1.setShortcut('Ctrl+U')
        newAction1.setStatusTip('Record Video Analysis Report')
        newAction1.triggered.connect(self.analysis_window)
        self.main.reporting.addAction(newAction1)

    # self.get_result_label = QtGui.QLabel(config_frame_p_main)
    # self.get_result_label.setText("Fetch Result of Recognition")
    # self.get_result_combo = QtGui.QComboBox()
    # self.get_result_combo.addItems(['No', 'Yes'])
    # self.get_result_combo.currentIndexChanged.connect(self.get_result_combo_change)

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
        # self.realmScroll = QtGui.QScrollArea(config_frame_p_main)
        # self.realmScroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        # self.realmScroll.setWidgetResizable(False)
        config_layout_p_main = pg.QtGui.QVBoxLayout()
        config_frame_p_main.setLayout(config_layout_p_main)
        advsetting.setWidget(config_frame_p_main)

        widgets = []

        title3 = u'&Camera'
        r_d_group_box_3 = QtGui.QGroupBox(title3)
        r_d_group_box_l_3 = QtGui.QVBoxLayout()

        self.camera_label = QtGui.QLabel(config_frame_p_main)
        self.camera_label.setText("Select Camera")
        self.camera_select = QtGui.QComboBox()
        cameras = self.detectNumAttachedCvCameras()
        list_of_camera = []
        for i in range(cameras):
            list_of_camera.append(str(i))
        self.camera_select.addItems(list_of_camera)
        self.camera_select.currentIndexChanged.connect(self.selectcamera)

        r_d_group_box_l_3.addWidget(self.camera_label)
        r_d_group_box_l_3.addWidget(self.camera_select)

        self.downsampleing_factor_label = QtGui.QLabel()
        self.downsampleing_factor_label.setText("Down Sampling Factor")
        self.downsampleing_factor = QtGui.QComboBox()
        self.downsampleing_factor.addItems(['1', '2', '3', '4', '5'])
        self.downsampleing_factor.currentIndexChanged.connect(self.downsampleing_factor_change)
        r_d_group_box_l_3.addWidget(self.downsampleing_factor_label)
        r_d_group_box_l_3.addWidget(self.downsampleing_factor)

        r_d_group_box_3.setLayout(r_d_group_box_l_3)
        config_layout_p_main.addWidget(r_d_group_box_3)

        title2 = u'&Detector'
        r_d_group_box_1 = QtGui.QGroupBox(title2)
        r_d_group_box_l_1 = QtGui.QVBoxLayout()

        self.face_detection_methods_lb = QtGui.QLabel(config_frame_p_main)
        self.face_detection_methods_lb.setText("Face Detection Method")
        self.face_detection_methods = QtGui.QComboBox()
        self.face_detection_methods.addItems(['mtcnn', 'dlib'])
        index = self.face_detection_methods.findText(self.configuration.detector_type, QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.face_detection_methods.setCurrentIndex(index)
        self.face_detection_methods.currentIndexChanged.connect(self.method_selection)
        r_d_group_box_l_1.addWidget(self.face_detection_methods_lb)
        r_d_group_box_l_1.addWidget(self.face_detection_methods)

        self.vertical_alignment_checkbox = QtGui.QCheckBox("Rotate Face")
        self.vertical_alignment_checkbox.setChecked(False)
        self.vertical_alignment_checkbox.stateChanged.connect(
            lambda: self.vertical_alignment_checkbox_change(self.vertical_alignment_checkbox))
        r_d_group_box_l_1.addWidget(self.vertical_alignment_checkbox)

        self.feature_point_checkbox = QtGui.QCheckBox("Show Feature Point")
        self.feature_point_checkbox.setChecked(False)
        self.feature_point_checkbox.stateChanged.connect(
            lambda: self.feature_point_checkbox_change(self.feature_point_checkbox))
        r_d_group_box_l_1.addWidget(self.feature_point_checkbox)

        self.addpadding_label = QtGui.QLabel()
        self.addpadding_label.setText("Add Padding for Rotation")
        self.addpadding = QtGui.QComboBox()
        self.addpadding.addItems(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90'])
        self.addpadding.currentIndexChanged.connect(self.addpadding_change)

        r_d_group_box_l_1.addWidget(self.addpadding_label)
        r_d_group_box_l_1.addWidget(self.addpadding)

        self.increase_box_size_lb = QtGui.QLabel()
        self.increase_box_size_lb.setText("Increase Bounding Box")
        self.sl = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sl.setMinimum(-20)
        self.sl.setMaximum(20)
        self.sl.setValue(0)
        self.sl.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sl.setTickInterval(10)
        self.sl.valueChanged.connect(self.sl_change)

        r_d_group_box_l_1.addWidget(self.increase_box_size_lb)
        r_d_group_box_l_1.addWidget(self.sl)

        r_d_group_box_1.setLayout(r_d_group_box_l_1)
        config_layout_p_main.addWidget(r_d_group_box_1)



        title1 = u'&Recognizer'

        r_d_h_group_box = pg.QtGui.QGroupBox(title1)
        r_d_h_group_box_l = QtGui.QHBoxLayout()

        r_d_group_box = QtGui.QGroupBox('')
        r_d_group_box_l = QtGui.QVBoxLayout()

        self.facerecognition_api_lb = QtGui.QLabel(config_frame_p_main)
        self.facerecognition_api_lb.setText("Face Recognition Method")
        self.facerecognition_api = QtGui.QComboBox()
        self.facerecognition_api.addItems(['facenet', 'nn', 'inception_v1', 'inception_v5', 'svm', 'svm_facenet'])
        index1 = self.facerecognition_api.findText(self.configuration.recognizer_type, QtCore.Qt.MatchFixedString)
        if index1 >= 0:
            self.facerecognition_api.setCurrentIndex(index1)
        self.facerecognition_api.currentIndexChanged.connect(self.api_selection)
        r_d_group_box_l.addWidget(self.facerecognition_api_lb)
        r_d_group_box_l.addWidget(self.facerecognition_api)

        self.no_of_recog_label = QtGui.QLabel()
        self.no_of_recog_label.setText("Number of Face to Recognize")
        self.no_of_recog = QtGui.QComboBox()
        self.no_of_recog.addItems(['1', '2', '3', '4', 'all'])
        self.no_of_recog.currentIndexChanged.connect(self.selection_recog_change)
        r_d_group_box_l.addWidget(self.no_of_recog_label)
        r_d_group_box_l.addWidget(self.no_of_recog)

        self.no_of_pred_label = QtGui.QLabel()
        self.no_of_pred_label.setText("Number of Predictions")
        self.no_of_pred = QtGui.QComboBox()
        self.no_of_pred.addItems(['1', '2', '3', '4', '5'])
        self.no_of_pred.currentIndexChanged.connect(self.number_of_prediction_change)

        r_d_group_box_l.addWidget(self.no_of_pred_label)
        r_d_group_box_l.addWidget(self.no_of_pred)
        r_d_group_box.setLayout(r_d_group_box_l)


        r_d_group_box_2 = QtGui.QGroupBox('')
        r_d_group_box_l_2 = QtGui.QVBoxLayout()

        self.accumulator_checkbox = QtGui.QCheckBox("Accumulate Recognition")
        self.accumulator_checkbox.setChecked(False)
        self.accumulator_checkbox.stateChanged.connect(
            lambda: self.accumulator_checkbox_change(self.accumulator_checkbox))
        r_d_group_box_l_2.addWidget(self.accumulator_checkbox)

        self.accumulator_label = QtGui.QLabel()
        self.accumulator_label.setText("No of frames to Accumulate")
        self.accumulator_pred = QtGui.QComboBox()
        self.accumulator_pred.addItems(['2', '10', '15', '20', '30'])
        self.accumulator_pred.currentIndexChanged.connect(self.accumulator_change)
        r_d_group_box_l_2.addWidget(self.accumulator_label)
        r_d_group_box_l_2.addWidget(self.accumulator_pred)
        r_d_group_box_2.setLayout(r_d_group_box_l_2)

        self.accumulator_wieghted_checkbox = QtGui.QCheckBox("Weighted Accumulator")
        self.accumulator_wieghted_checkbox.setChecked(False)
        self.accumulator_wieghted_checkbox.stateChanged.connect(
            lambda: self.accumulator_wieghted_checkbox_change(self.accumulator_wieghted_checkbox))
        r_d_group_box_l_2.addWidget(self.accumulator_wieghted_checkbox)


        r_d_h_group_box_l.addWidget(r_d_group_box)
        r_d_h_group_box_l.addWidget(r_d_group_box_2)
        r_d_h_group_box.setLayout(r_d_h_group_box_l)
        config_layout_p_main.addWidget(r_d_h_group_box)


        title4 = u'&Capture Face'
        r_d_group_box_4 = QtGui.QGroupBox(title4)
        r_d_group_box_l_4 = QtGui.QVBoxLayout()

        self.video_codec_label = QtGui.QLabel(config_frame_p_main)
        self.video_codec_label.setText("Video Codec")
        self.video_codec_combobox = QtGui.QComboBox()
        self.video_codec_combobox.addItems(['XVID','X264'])
        self.video_codec_combobox.currentIndexChanged.connect(self.choose_video_codec)
        r_d_group_box_l_4.addWidget(self.video_codec_label)
        r_d_group_box_l_4.addWidget(self.video_codec_combobox)

        self.capture_face_number_label = QtGui.QLabel(config_frame_p_main)
        self.capture_face_number_label.setText("Capture Faces")
        self.capture_face_number_label_combobox = QtGui.QComboBox()
        self.capture_face_number_label_combobox.addItems(['Near to Camera', 'All Faces'])
        self.capture_face_number_label_combobox.currentIndexChanged.connect(self.capture_face_number_change)
        r_d_group_box_l_4.addWidget(self.capture_face_number_label)
        r_d_group_box_l_4.addWidget(self.capture_face_number_label_combobox)

        r_d_group_box_4.setLayout(r_d_group_box_l_4)
        config_layout_p_main.addWidget(r_d_group_box_4)


        # title6 = u'&Reporting'
        # r_d_group_box_6 = QtGui.QGroupBox(title6)
        # r_d_group_box_l_6 = QtGui.QVBoxLayout()
        #
        # self.get_result_label = QtGui.QLabel(config_frame_p_main)
        # self.get_result_label.setText("Fetch Result of Recognition")
        # self.get_result_combo = QtGui.QComboBox()
        # self.get_result_combo.addItems(['No', 'Yes'])
        # self.get_result_combo.currentIndexChanged.connect(self.get_result_combo_change)
        #
        # self.button_show_reprt = QtGui.QPushButton('Show Report', config_frame_p_main)
        # self.button_show_reprt.clicked.connect(self.button_show_reprt_click)
        # r_d_group_box_l_6.addWidget(self.get_result_label)
        # r_d_group_box_l_6.addWidget(self.get_result_combo)
        # r_d_group_box_l_6.addWidget(self.button_show_reprt)
        #
        # r_d_group_box_6.setLayout(r_d_group_box_l_6)
        # config_layout_p_main.addWidget(r_d_group_box_6)

        # self.no_of_pred_label = QtGui.QLabel(config_frame_p_main)
        # self.no_of_pred_label.setText("Number of Predictions")
        # self.no_of_pred = QtGui.QComboBox()
        # self.no_of_pred.addItems(['1', '2', '3', '4', '5'])
        # self.no_of_pred.currentIndexChanged.connect(self.number_of_prediction_change)
        #
        # widgets.append(self.no_of_pred_label)
        # widgets.append(self.no_of_pred)
        #
        # self.face_detection_methods_lb = QtGui.QLabel(config_frame_p_main)
        # self.face_detection_methods_lb.setText("Face Detection Method")
        # self.face_detection_methods = QtGui.QComboBox()
        # self.face_detection_methods.addItems(['mtcnn', 'dlib'])
        # index = self.face_detection_methods.findText(self.configuration.detector_type, QtCore.Qt.MatchFixedString)
        # if index >= 0:
        #     self.face_detection_methods.setCurrentIndex(index)
        # self.face_detection_methods.currentIndexChanged.connect(self.method_selection)
        #
        # widgets.append(self.face_detection_methods_lb)
        # widgets.append(self.face_detection_methods)
        #
        # self.facerecognition_api_lb = QtGui.QLabel(config_frame_p_main)
        # self.facerecognition_api_lb.setText("Face Recognition Method")
        # self.facerecognition_api = QtGui.QComboBox()
        # self.facerecognition_api.addItems(['facenet', 'nn', 'inception_v1', 'inception_v5','svm','svm_facenet'])
        # index1 = self.facerecognition_api.findText(self.configuration.recognizer_type, QtCore.Qt.MatchFixedString)
        # if index1 >= 0:
        #     self.facerecognition_api.setCurrentIndex(index1)
        # # self.facerecognition_api.currentText(self.configuration.recognizer_type)
        # self.facerecognition_api.currentIndexChanged.connect(self.api_selection)
        #
        # widgets.append(self.facerecognition_api_lb)
        # widgets.append(self.facerecognition_api)
        #
        # self.camera_label = QtGui.QLabel(config_frame_p_main)
        # self.camera_label.setText("Select Camera")
        # self.camera_select = QtGui.QComboBox()
        # cameras = self.detectNumAttachedCvCameras()
        # list_of_camera = []
        # for i in range(cameras):
        #     list_of_camera.append(str(i))
        # self.camera_select.addItems(list_of_camera)
        # self.camera_select.currentIndexChanged.connect(self.selectcamera)
        #
        # widgets.append(self.camera_label)
        # widgets.append(self.camera_select)
        #
        # # self.frame_rate_label = QtGui.QLabel(config_frame_p_main)
        # # self.frame_rate_label.setText("Frame Rate Per Sec")
        # # self.frame_rate_combobox = QtGui.QComboBox()
        # # self.frame_rate_combobox.addItems(['2', '5', '10', '15', '20', '30'])
        # # self.frame_rate_combobox.currentIndexChanged.connect(self.frame_rate_combobox_change)
        #
        # # widgets.append(self.frame_rate_label)
        # # widgets.append(self.frame_rate_combobox)
        #
        # self.video_codec_label = QtGui.QLabel(config_frame_p_main)
        # self.video_codec_label.setText("Video Codec")
        # self.video_codec_combobox = QtGui.QComboBox()
        # self.video_codec_combobox.addItems(['XVID','X264'])
        # self.video_codec_combobox.currentIndexChanged.connect(self.choose_video_codec)
        #
        # widgets.append(self.video_codec_label)
        # widgets.append(self.video_codec_combobox)
        #
        # self.capture_face_number_label = QtGui.QLabel(config_frame_p_main)
        # self.capture_face_number_label.setText("Capture Faces")
        # self.capture_face_number_label_combobox = QtGui.QComboBox()
        # self.capture_face_number_label_combobox.addItems(['Near to Camera', 'All Faces'])
        # self.capture_face_number_label_combobox.currentIndexChanged.connect(self.capture_face_number_change)
        #
        # widgets.append(self.capture_face_number_label)
        # widgets.append(self.capture_face_number_label_combobox)
        #
        # self.get_result_label = QtGui.QLabel(config_frame_p_main)
        # self.get_result_label.setText("Fetch Result of Recognition")
        # self.get_result_combo = QtGui.QComboBox()
        # self.get_result_combo.addItems(['No', 'Yes'])
        # self.get_result_combo.currentIndexChanged.connect(self.get_result_combo_change)
        #
        # widgets.append(self.get_result_label)
        # widgets.append(self.get_result_combo)
        #
        # self.feature_point_checkbox = QtGui.QCheckBox("Show Feature Point")
        # self.feature_point_checkbox.setChecked(False)
        # self.feature_point_checkbox.stateChanged.connect(lambda: self.feature_point_checkbox_change(self.feature_point_checkbox))
        # widgets.append(self.feature_point_checkbox)
        # widgets.append(None)
        #
        # self.vertical_alignment_checkbox = QtGui.QCheckBox("Align Face Vertically")
        # self.vertical_alignment_checkbox.setChecked(False)
        # self.vertical_alignment_checkbox.stateChanged.connect(
        #     lambda: self.vertical_alignment_checkbox_change(self.vertical_alignment_checkbox))
        # widgets.append(self.vertical_alignment_checkbox)
        # widgets.append(None)
        #
        #
        # self.button_show_reprt = QtGui.QPushButton('Show Report', config_frame_p_main)
        # self.button_show_reprt.clicked.connect(self.button_show_reprt_click)
        #
        # widgets.append(None)
        # widgets.append(self.button_show_reprt)

        row = 0
        col = 1
        for widget in widgets:
            if widget is not None:

                config_layout_p_main.addWidget(widget, row, col - 1)

                if col % 2 == 0:
                    row += 1
                    col = 1
                else:
                    col += 1
            else:
                col += 1

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

    def accumulator_change(self, i):
        value = self.accumulator_pred.currentText()
        self.main.change_config('accumulator_frame_count', value)

    def downsampleing_factor_change(self, i):
        value = self.downsampleing_factor.currentText()
        self.main.change_config('down_sampling_factor', value)

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