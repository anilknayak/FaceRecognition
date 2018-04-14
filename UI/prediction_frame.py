from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import cv2
import numpy as np


class Prediction(QtGui.QDockWidget):
    def __init__(self, mainWindow):
        super(Prediction, self).__init__()
        # self.prediction_dock = QtGui.QDockWidget("Predictions", mainWindow)
        # self.prediction_dock.setGeometry(0, 0, 1000, 600)
        # # self.prediction_dock.setFixedWidth(200)
        # self.prediction_dock.setWindowIcon(QtGui.QIcon('./UI/images.png'))
        # self.prediction_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        # mainWindow.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.prediction_dock)
        # mainWindow.viewMenu.addAction(self.prediction_dock.toggleViewAction())
        self.main = mainWindow
        # self.s1 = pg.QtGui.QScrollBar()
        # self.s1.setMaximum(255)
        self.pix = True


        self.config_frame_p_main = pg.QtGui.QFrame()
        self.config_layout_p_main = pg.QtGui.QGridLayout()
        self.config_frame_p_main.setLayout(self.config_layout_p_main)
        # self.prediction_dock.setWidget(self.config_frame_p_main)

        self.setWidget(self.config_frame_p_main)
        self.setWindowTitle("Predictions")
        self.setWindowIcon(QtGui.QIcon("./UI/images.png"))
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        # self.scrollArea = QtGui.QScrollArea(self.config_frame_p_main)
        # self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        # self.scrollArea.setWidgetResizable(True)
        # self.scrollArea.setObjectName("scrollArea")
        # self.scrollArea.setEnabled(True)
        # self.scrollArea.setWidgetResizable(True)
        # self.scrollAreaWidgetContents = QtGui.QWidget(self.scrollArea)
        # self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 380, 247))
        # self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.frames_obj = []
        self.layout_obj = []
        self.frames = []
        self.number_of_faces = 0
        self.number_of_prediction = 1
        self.total_number_of_rows_in_frame = 6

        self.prepare_prediction_frame_onchange()

    def changeEvent(self, event):
        self.change_state()

    def change_state(self):
        row, col = self.main.getPredictionNumbers()
        screen = QtGui.QDesktopWidget().availableGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()

        width_to_set = int(row) * 230
        height_to_set = (int(col) + 1) * 280

        if height_to_set > self.screen_height - 100:
            height_to_set = self.screen_height - 100

        if width_to_set > self.screen_width - 100:
            width_to_set = self.screen_width - 100

        if self.isFloating():
            self.setFixedWidth(width_to_set)
            self.setFixedHeight(height_to_set)
            self.adjustSize()
            self.update()
        else:
            self.setFixedWidth(width_to_set)
            self.setFixedHeight(height_to_set)
            self.adjustSize()
            self.update()



    def setNumberOfFaces(self, no):
        self.number_of_faces = no

    def setNumberOfPrediction(self, no):
        self.number_of_prediction = no

    def prepare_deck(self):
        ''
        # self.prepare_pred_frames(mainWindow)
        # if self.prediction_dock.isFloating():
        #     # row, col = self.main.getPredictionNumbers()
        #     self.prediction_dock.setFixedWidth(300)
        # else:
        #     self.prediction_dock.setFixedWidth(150)

    def prepare_prediction_frame_onchange(self):
        self.prepare_deck()
        if self.number_of_faces > len(self.frames_obj):
            for i in range(len(self.frames_obj), self.number_of_faces, 1):
                self.create_display_face_column_frames(i)

        self.create_display_faces_rows()

    def create_display_face_column_frames(self, index):
        config_frame = pg.QtGui.QFrame()
        config_layout = pg.QtGui.QGridLayout()
        config_frame.setLayout(config_layout)
        self.config_layout_p_main.addWidget(config_frame, 0, index)
        self.frames_obj.append(config_frame)
        self.layout_obj.append(config_layout)
        self.frames.append([])

    def create_display_faces_rows(self):
        for index_obj in range(0, len(self.frames_obj), 1):
            frame = self.frames_obj[index_obj]
            layout = self.layout_obj[index_obj]

            if not len(self.frames[index_obj]) == self.total_number_of_rows_in_frame:
                display_frames_temp = []
                for row in range(0, self.total_number_of_rows_in_frame * 2, 2):
                    display_frame = []

                    label = QtGui.QLabel(frame)
                    layout.addWidget(label, (row), 0)

                    if not row == 0:
                        pred_image = QtGui.QLabel()
                        layout.addWidget(pred_image, (row + 1), 0)
                    else:
                        pred_img_frame = pg.GraphicsView(frame)
                        pred_img_frame.setBackground(None)
                        pred_img_frame.setAspectLocked(True)
                        pred_image = pg.ImageItem()
                        pred_img_frame.addItem(pred_image)
                        layout.addWidget(pred_img_frame, (row + 1), 0)

                    display_frame.append(pred_image)
                    display_frame.append(label)
                    display_frames_temp.append(display_frame)

                self.frames[index_obj] = display_frames_temp


    def display_predicted_faces(self, faces):
        # print(self.frames)
        if faces is not None and len(faces) > 0:
            for j in range(0, len(faces), 1):
                face = faces[j]
                frame = self.frames[j] # now this frame is a column frame which has mutiple rows
                face_image = face.image
                images = face.images_pred
                titles = face.labels_pred
                paths = face.image_paths
                probs = face.prob_pred

                correct_acc = None

                if face.recognition_stat is not None and len(face.recognition_stat)>0:
                    correct_acc = face.recognition_stat

                self.publish_single_face_detected(face_image, images, titles,paths, correct_acc , probs, frame)

            if len(faces) < len(self.frames):
                for j in range(len(faces), len(self.frames), 1):
                    self.clear_frames(j)

    def clear_all_frames(self):
        self.prepare_deck()
        for j in range(len(self.frames)):
            self.clear_frames(j)

    def clear_frames(self, j):
        frame = self.frames[j]
        for i in range(len(frame)):
            image_frame = frame[i][0]
            title_frame = frame[i][1]
            image_frame.clear()
            title_frame.setText('')

    def publish_single_face_detected(self, face_image, images, titles,paths,correct_acc, probs, frame):
        size = 120
        image_frame = frame[0][0]
        title_frame = frame[0][1]
        resized_face1 = cv2.resize(np.squeeze(face_image), (size, size), interpolation=cv2.INTER_CUBIC)
        title_frame.setText("Captured Face")
        image_frame.setImage(resized_face1)

        # if self.pix:
        #     try:
        #         height, width, channel = resized_face1.shape
        #         bytesPerLine = 3 * width
        #         pixmap = pg.QtGui.QImage(resized_face1, width, height, bytesPerLine, pg.QtGui.QImage.Format_RGB888)
        #     except:
        #         height, width = resized_face1.shape
        #         # bytesPerLine = 3 * width
        #         p = pg.QtGui.QImage(resized_face1, width, height, pg.QtGui.QImage.Format_RGB888)
        #         pixmap = pg.QtGui.QPixmap.fromImage(p)
        #     pix = QtGui.QPixmap(pixmap)
        #     image_frame.setPixmap(pix)
        # else:

        flag = False
        if correct_acc is not None:
            flag = True

        for row in range(self.number_of_prediction):
            image_frame = frame[row+1][0]
            title_frame = frame[row+1][1]
            if flag:
                title_text = titles[row]+ " : " + str(probs[row]) + "[ " + correct_acc[row] + " ]"
            else:
                title_text = titles[row] + " : " + str(probs[row])
            title_frame.setText(title_text)
            image_path = paths[row]
            if self.pix:
                splash_pix = pg.QtGui.QPixmap(image_path)
                splash_pix = splash_pix.scaledToHeight(size)
                splash_pix = splash_pix.scaledToWidth(size)
                image_frame.setPixmap(splash_pix)
            else:
                img = cv2.imread(image_path,1)
                resized_face = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                # resized_face = cv2.resize(np.squeeze(images[row]), (size, size), interpolation=cv2.INTER_CUBIC)
                image_frame.setImage(resized_face)

        if self.number_of_prediction < 5:
            for row in range(self.number_of_prediction, 5, 1):
                image_frame = frame[row + 1][0]
                title_frame = frame[row + 1][1]
                image_frame.clear()
                title_frame.setText('')


    def prepare_pred_frames(self,mainWindow):
        mainWindow.frames = []

        # for j in range(mainWindow.number_of_faces_found_in_image):
        #     config_frame = pg.QtGui.QFrame()
        #     config_layout = pg.QtGui.QGridLayout()
        #     config_frame.setLayout(config_layout)
        #     prediction.setWidget(config_frame, 0, j)
        #
        #     for i in range(mainWindow.recognize.configuration.number_of_prediction + 1):
        #         pred_img_frame = pg.GraphicsLayoutWidget(config_frame)
        #         prediction_view = pred_img_frame.addViewBox()

        self.config_frame_p_main = pg.QtGui.QFrame()
        self.config_layout_p_main = pg.QtGui.QGridLayout()
        self.config_frame_p_main.setLayout(self.config_layout_p_main)
        self.prediction_dock.setWidget(self.config_frame_p_main)

        for j in range(mainWindow.number_of_faces_found_in_image):
            print("preparing")
            # per_person_frame = pg.GraphicsLayoutWidget(mainWindow)
            config_frame = pg.QtGui.QFrame()
            config_layout = pg.QtGui.QGridLayout()
            config_frame.setLayout(config_layout)
            self.config_layout_p_main.addWidget(config_frame, 0, j)
            frames_s = []

            for i in range(0, mainWindow.commuter.get_no_of_prediction()+1):
                frame = []
                # pred_img_frame = pg.GraphicsLayoutWidget(config_frame)
                # prediction_view = pred_img_frame.addViewBox()
                # prediction_view.setBackgroundColor((255, 255, 255, 255))
                # prediction_view.setAspectLocked(True)
                # prediction_view.invertY()
                # prediction_view.invertX()
                # prediction_view.setMouseEnabled(False, False)
                # label = QtGui.QLabel(config_frame)
                # pred_image = pg.ImageItem()
                # prediction_view.addItem(pred_image)
                # frame.append(pred_image)
                # frame.append(label)
                # config_layout.addWidget(pred_img_frame, (i + 1), 0)
                # config_layout.addWidget(label, (i + 1), 1)
                # frames_s.append(frame)

                pred_img_frame = pg.GraphicsView(config_frame)
                pred_img_frame.setBackground(None)
                pred_img_frame.setAspectLocked(True)
                label = QtGui.QLabel(config_frame)
                pred_image = pg.ImageItem()
                pred_img_frame.addItem(pred_image)
                frame.append(pred_image)
                frame.append(label)
                config_layout.addWidget(pred_img_frame, (i + 1), 0)
                config_layout.addWidget(label, (i + 1), 1)
                frames_s.append(frame)


            mainWindow.frames.append(frames_s)
        # print(mainWindow.frames)