from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import cv2
from UI import file_dialog as fd
from UI import advance_setting as ads
class Setting:
    def __init__(self, mainWindow, configuration):
        self.main = mainWindow
        self.configuration = configuration
        self.setting_dock = QtGui.QDockWidget("Setting", self.main)
        self.setting_dock.setGeometry(0, 0, 1000, 600)
        self.setting_dock.setWindowIcon(QtGui.QIcon('images.png'))
        self.setting_dock.setAllowedAreas(QtCore.Qt.TopDockWidgetArea)
        self.main.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.setting_dock)
        self.main.editMenu.addAction(self.setting_dock.toggleViewAction())
        self.prepare_setting_frames(self.setting_dock)

    def prepare_setting_frames(self,setting):
        config_frame_p_main = pg.QtGui.QFrame()
        config_layout_p_main = pg.QtGui.QGridLayout()
        config_frame_p_main.setLayout(config_layout_p_main)
        setting.setWidget(config_frame_p_main)

        widgets = []

        '''New Configurations'''
        self.load_data_medium_combobox_label = QtGui.QLabel(config_frame_p_main)
        self.load_data_medium_combobox_label.setText("Select Source")
        self.load_data_medium_combobox = QtGui.QComboBox()
        self.load_data_medium_combobox.addItems(['Camera', 'Video File'])
        self.load_data_medium_combobox.currentIndexChanged.connect(self.load_data_medium_combobox_change)
        self.source = 'Camera'

        widgets.append(self.load_data_medium_combobox_label)
        widgets.append(self.load_data_medium_combobox)

        self.action_combobox_label = QtGui.QLabel(config_frame_p_main)
        self.action_combobox_label.setText("Select Action")
        self.action_combobox = QtGui.QComboBox()
        self.action_combobox.addItems(['Recognize', 'Display',  'Capture'])
        self.action_combobox.currentIndexChanged.connect(self.action_combobox_change)

        widgets.append(self.action_combobox_label)
        widgets.append(self.action_combobox)

        self.button_start_capturing_data = QtGui.QPushButton('Start', config_frame_p_main)
        self.button_start_capturing_data.clicked.connect(self.start_reading_from_source_onlick)

        widgets.append(None)
        widgets.append(self.button_start_capturing_data)

        self.video_file_name_label = QtGui.QLabel(config_frame_p_main)
        self.video_file_name_label.setText("")
        self.change_file_btn = QtGui.QPushButton('Change Video File', config_frame_p_main)
        self.change_file_btn.clicked.connect(self.load_data_medium_combobox_change)
        self.change_file_btn.hide()

        widgets.append(self.video_file_name_label)
        widgets.append(self.change_file_btn)


        self.video_file_name = QtGui.QLabel(config_frame_p_main)
        self.video_file_name.setText("")

        widgets.append(self.video_file_name)
        widgets.append(None)

        '''New Configurations'''

        row = 0
        col = 0
        for widget in widgets:
            if widget is not None:
                config_layout_p_main.addWidget(widget, row, col)
                col += 1

    def load_data_medium_combobox_change(self, i):
        self.source = self.load_data_medium_combobox.currentText()
        if self.source == "Video File":
            self.openFileNameDialog()
            self.set_video_file_name_label(self.filename)
            self.change_file_btn.show()
        else:
            self.set_video_file_name_label(None)
            self.change_file_btn.hide()

    def openFileNameDialog(self):
        fd_obj = fd.OpenFileDialog()
        fd_obj.initUI('video')
        self.filename = fd_obj.filename
        self.main.video_file = self.filename

    def set_video_file_name_label(self, filename):
        if filename is not None:
            self.video_file_name_label.setText('Video File Name')
            f = str(filename).split("/")[-1]
            self.video_file_name.setText(str(f))
        else:
            self.video_file_name_label.setText('')
            self.video_file_name.setText('')

    def action_combobox_change(self):
        self.main.change_config('action', str(self.action_combobox.currentText()))

    def toggle_start_button(self):
        self.button_start_capturing_data.setText('Start')
        self.main.stop_capturing_source()

    def start_reading_from_source_onlick(self):
        button_text = self.button_start_capturing_data.text()
        self.main.change_config('action', str(self.action_combobox.currentText()))
        self.source = self.load_data_medium_combobox.currentText()

        if button_text == 'Start':
            if self.source == 'Camera':
                self.main.start_camera = True
                self.main.video_file = None
                self.main.source = self.source
                self.main.start_capturing_source()

            elif self.source == 'Video File':
                self.main.start_camera = False
                self.main.source = self.source
                self.main.start_capturing_source()

            self.button_start_capturing_data.setText('Stop')
        else:
            self.main.stop_capturing_source()
            self.button_start_capturing_data.setText('Start')

    def start_camera_button(self):
        self.main.start_camera = True
        self.main.video_file = None
        self.main.source = self.source
        self.main.start_capturing_source()




