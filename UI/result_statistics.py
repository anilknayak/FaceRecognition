from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from UI import file_dialog as fd
class OpenResultStatistics(QtGui.QMainWindow):
    def __init__(self, base_dir, parent=None):
        super(OpenResultStatistics, self).__init__(parent)
        self.base_dir = base_dir
        screen = QtGui.QDesktopWidget().availableGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.title = 'Choose Video Files'
        self.left = self.screen_width // 2 - 300
        self.top = self.screen_height // 2 - 200
        self.width = 640
        self.height = 480

        frame = pg.QtGui.QFrame()
        layout = pg.QtGui.QGridLayout()
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        frame.setLayout(layout)

        # self.Preferences_dock = QtGui.QDockWidget("Preferences", self)
        # self.Preferences_dock.setWindowIcon(QtGui.QIcon('images.png'))
        # self.Preferences_dock.setAllowedAreas(QtCore.Qt.TopDockWidgetArea)
        # self.main.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.Preferences_dock)
        # self.main.viewMenu.addAction(self.Preferences_dock.toggleViewAction())
        self.video_number = 1
        self.filename_video_analysis = None
        self.report_for = None
        self.report_type = None
        self.prepare_preference_frames(layout)
        self.prepare_display_frames(layout)
        self.setCentralWidget(frame)
        self.setWindowTitle("Report")
        self.setWindowIcon(QtGui.QIcon('images.png'))
        self.setGeometry(self.left, self.top, self.width, self.height)

    def prepare_display_frames(self, layout):
        self.graphics_window = pg.GraphicsView(self)
        self.graphics_window.setBackground(None)
        self.report_image = pg.ImageItem()  # border='w'
        self.graphics_window.addItem(self.report_image)
        layout.addWidget(self.graphics_window, 0, 2, 1, 1)



    def prepare_preference_frames(self, layout):
        widgets = []
        config_frame_p_main = pg.QtGui.QFrame()
        config_layout_p_main = pg.QtGui.QGridLayout()
        config_frame_p_main.setLayout(config_layout_p_main)
        layout.addWidget(config_frame_p_main, 0, 1, 1, 1)

        self.report_for_combobox_label = QtGui.QLabel(config_frame_p_main)
        self.report_for_combobox_label.setText("Report For")
        self.report_for_combobox = QtGui.QComboBox()
        self.report_for_combobox.addItems(
            ['CNN', 'Inception V1', 'Inception V5', 'SVM', 'FACENET', 'SVM_FACENET', 'All Comparision'])
        self.report_for_combobox.currentIndexChanged.connect(self.load_report_for_combobox_change)
        self.report_for = 'CNN'

        widgets.append(self.report_for_combobox_label)
        widgets.append(self.report_for_combobox)

        self.report_type_combobox_label = QtGui.QLabel(config_frame_p_main)
        self.report_type_combobox_label.setText("Report Type")
        self.report_type_combobox = QtGui.QComboBox()
        self.report_type_combobox.addItems(
            ['Data Details', 'Loss', 'Validation Accuracy', 'Video Analysis', 'Comparision'])
        self.report_type_combobox.currentIndexChanged.connect(self.load_report_type_combobox_change)
        self.report_type = 'Data Details'

        widgets.append(self.report_type_combobox_label)
        widgets.append(self.report_type_combobox)

        self.combobox_label = QtGui.QLabel(config_frame_p_main)
        self.combobox_label.setText("Video Number")
        self.combobox = QtGui.QComboBox()
        self.combobox.addItems(['1', '2', '3', '4', '5'])
        self.combobox.currentIndexChanged.connect(self.combobox_change)

        widgets.append(self.combobox_label)
        widgets.append(self.combobox)

        self.button_show_video = QtGui.QPushButton('Choose Video', config_frame_p_main)
        self.button_show_video.clicked.connect(self.load_data_change)

        widgets.append(None)
        widgets.append(self.button_show_video)

        # self.video_type_combobox_label = QtGui.QLabel(config_frame_p_main)
        # self.video_type_combobox_label.setText("Video Analysis Files")
        # self.video_type_combobox = QtGui.QComboBox()
        # self.video_type_combobox.addItems(
        #     ['Data Details', 'Loss', 'Validation Accuracy', 'Video Analysis', 'Comparision'])
        # self.video_type_combobox.currentIndexChanged.connect(self.video_type_combobox_chamge)
        # self.report_type = 'Data Details'
        #
        # widgets.append(self.report_type_combobox_label)
        # widgets.append(self.report_type_combobox)

        self.button_show = QtGui.QPushButton('Show', config_frame_p_main)
        self.button_show.clicked.connect(self.button_show_onlick)

        widgets.append(None)
        widgets.append(self.button_show)

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

    def combobox_change(self, i):
        self.video_number = self.combobox.currentText()

    def load_data_change(self):
        self.openFileNameDialog()

    def openFileNameDialog(self):
        fd_obj = fd.OpenFileDialog()
        fd_obj.initUI('pickle')
        self.filename_video_analysis = fd_obj.filename

    def load_report_type_combobox_change(self, i):
        self.report_type = self.report_type_combobox.currentText()

    def load_report_for_combobox_change(self, i):
        self.report_for = self.report_for_combobox.currentText()

    def button_show_onlick(self):
        self.prepare_dir()
        key, flag = self.prepare_file()

        if flag:
            if key == 'data':
                self.display_graph_data(key)

            elif key == 'loss':
                self.display_loss_validation_data(key, 'Loss Graph', 'Iteration', 'Cross Entropy Loss')

            elif key == 'evaluation':
                self.display_loss_validation_data(key, 'Validation Accuracy Graph', 'Iteration',
                                                  'Iteration Accuracy')
            elif key == 'video_analysis':
                self.view_video_analysis('video_analysis')



    def view_video_analysis(self, file):

        self.file = os.path.join(self.dir, "Video_"+str(self.video_number)+".pickle")
        self.load_file()
        self.display_video_analysis('video_analysis')

        # if self.filename_video_analysis is not None:
        #     self.file = os.path.join(self.dir, self.filename_video_analysis)
        #     self.load_file()
        #     self.display_video_analysis('video_analysis')
        # else:
        #     print('file is not correct')

    def prepare_dir(self):
        if self.report_for == 'CNN':
            self.dir = os.path.join(self.base_dir, 'result/cnn')
        elif self.report_for == 'Inception V1':
            self.dir = os.path.join(self.base_dir, 'result/inceptionv1')
        elif self.report_for == 'Inception V5':
            self.dir = os.path.join(self.base_dir, 'result/inceptionv5')
        elif self.report_for == 'SVM':
            self.dir = os.path.join(self.base_dir, 'result/svm')
        elif self.report_for == 'FACENET':
            self.dir = os.path.join(self.base_dir, 'result/facenet')
        elif self.report_for == 'SVM_FACENET':
            self.dir = os.path.join(self.base_dir, 'result/svm_facenet')

    def prepare_file(self):
        key = None
        flag = False
        if self.report_type == 'Data Details':
            self.file = os.path.join(self.dir, 'data_stat.pickle')
            self.load_file()
            flag = True
            key = 'data'
        elif self.report_type == 'Loss':
            self.file = os.path.join(self.dir, 'loss.pickle')
            self.load_file()
            flag = True
            key = 'loss'
        elif self.report_type == 'Validation Accuracy':
            self.file = os.path.join(self.dir, 'evaluation.pickle')
            self.load_file()
            flag = True
            key = 'evaluation'
        elif self.report_type == 'Video Analysis':
            key = 'video_analysis'
            flag = True

        return key, flag

    def load_file(self):
        print(self.file)
        label_pickle = open(self.file, "rb")
        self.label_pickle_dict = pickle.load(label_pickle)
        label_pickle.close()

    def display_graph_data(self, key):
        dicts = self.label_pickle_dict[key]
        inital = dicts['initial']
        augment = dicts['augment']

        print(inital)
        sorted_dict_initial = dict(sorted(inital.items(), key=lambda x: x[0]))
        sorted_dict_augment = dict(sorted(augment.items(), key=lambda x: x[0]))

        x_inital = []
        y_inital = []
        for a in sorted_dict_initial.keys():
            value = sorted_dict_initial[a]

            x_inital.append(a)
            y_inital.append(value)

        plt.bar(x_inital, y_inital)
        plt.xlabel('Person Name')
        plt.ylabel('Face Image Collected')
        plt.title('Initial Collection of data from person')
        plt.grid(True)
        plt.show()

        x_inital = []
        y_inital = []
        for a in sorted_dict_augment.keys():
            value = sorted_dict_augment[a]

            x_inital.append(a)
            y_inital.append(value)

        plt.bar(x_inital, y_inital)
        plt.xlabel('Person Name')
        plt.ylabel('Face Image Augmented')
        plt.title('Augmented data from person')
        plt.grid(True)
        plt.show()

    def display_loss_validation_data(self, key, title, xlabel, ylabel):
        list = self.label_pickle_dict[key]
        number = len(list)

        x_inital = np.linspace(1, number, number)
        y_inital = list

        plt.plot(x_inital, y_inital)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def display_video_analysis(self, key):
        data = self.label_pickle_dict[key]

        total_frame = data['total_frame']
        total_frame_face_detected = data['total_frame_face_detected']
        total_frame_face_recognized = data['total_frame_face_recognized']
        # method = data['method']

        frame_wise_recognition_list = data['frame_wise_recognition_list']

        unique_label = []
        number_of_frame_detected = {}
        for frame_avalisys_dict in frame_wise_recognition_list:
            label = frame_avalisys_dict['label']
            prob = frame_avalisys_dict['prob']
            frame_number = frame_avalisys_dict['frame_number']

            if not label in unique_label:
                unique_label.append(label)

            if label in number_of_frame_detected.keys():
                number_of_frame_detected[label] = number_of_frame_detected[label] + 1
            else:
                number_of_frame_detected[label] = 1

        recognized = []
        for label in unique_label:
            recognized.append(number_of_frame_detected[label])

        if not len(recognized) == len(unique_label):
            if len(recognized) - len(unique_label) > 0:
                for i in range(len(recognized) - len(unique_label)):
                    unique_label.append('None')

            if len(unique_label) - len(recognized) > 0:
                for i in range(len(unique_label) - len(recognized)):
                    recognized.append(0)


        x_inital = ['Total Frames','Detected Frames','Recognized Frames']
        y_inital = [total_frame,total_frame_face_detected,total_frame_face_recognized]
        plt.bar(x_inital, y_inital)
        plt.xlabel('Frame Analysis')
        plt.ylabel('Number Faces in Frame Analysis')
        plt.title('Frame Wise Analysis Report for Particular Videos')
        plt.grid(False)
        plt.show()

        print("Frame : ", y_inital)

        x_inital = unique_label
        y_inital = recognized
        plt.bar(x_inital, y_inital)
        plt.xlabel('Persons Recognized Name')
        plt.ylabel('Number of Time thet Person Recognized')
        plt.title('Recognition Report for Video Analysis')
        plt.grid(False)
        plt.show()

        print("Recognition : ", y_inital)


    def display_graph(self, key):
        print(self.label_pickle_dict[key])

