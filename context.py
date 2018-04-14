import os
import tensorflow as tf
from data import data_loader
from configuration import configuration as config
from configuration import information as info
from trained_model import model
from facedetection import detect
from facerecognition import recognize
from postprocessing import postprocessing as postprocess
from preprocessing import preprocessing as preprocess
from augmentation_data import augmentation as augment1
from preprocessing import data_separation as dataseparation
from configuration import nn_model_new as nnmodel
from utils import batch as pb
from result import export_result as rs

class Context:
    def __init__(self, action, debug_flag=False):
        self.debug_flag = debug_flag
        self.base_directory = os.path.dirname(__file__)
        # Conditions for Loading Context
        # Either GUI, TRAINING
        print('Starting the Face Recognition Module Context Creation for ', action)

        print('Creating Configuration Module in Context')
        # Load Configuration for UI and training
        self.configuration = config.Configuration(self.base_directory, debug_flag)
        self.configuration.load_configuration(os.path.join(os.path.dirname(__file__), "configuration/application/app.config"))

        print('Creating Tensorflow in Session')
        # Load tensorflow session for training
        # self.config = tf.ConfigProto()
        # self.config.gpu_options.allow_growth = False
        # self.sess = tf.InteractiveSession(config=self.config)

        print('Creating Data Module in Context')
        # Load Data Module which will do everything about data Reading, Writing, Separating data
        self.data_loader = data_loader.DataLoader(self.base_directory, self.configuration)

        print('Creating Model Module in Context')
        # Load Model module to store and restore model after training or testing time
        self.model_loader = model.Model(self.base_directory, self.configuration)

        print('Creating Information Module in Context')
        # Load Info Module to see the information about the module
        self.info = info.Information()

        print('Creating Pre Processing Module in Context')
        # Load Preprocessing Module will do the preprocessing
        self.preprocess = preprocess.Preprocessing(self.configuration)

        self.resultwritter = rs.ResultWriter(self.base_directory, self.configuration)

        if action == "TRAIN":
            print('Creating Data Augmentation Module in Context')
            # Load DataAugumentation Module will do the data augumentation
            self.augment = augment1.Augmentation(self.configuration)

            print('Creating data separation Module in context')
            self.data_separation = dataseparation.DataSeparation(self.configuration)

            print('Creating Model Preparation Module in Context')
            self.model_preparation = nnmodel.NNModel(self.base_directory, self.configuration)

            print('Creating Batch Module in Context')
            self.batch_prepare = pb.Prepare_Batch()

            print('Creating Evaluation Module in Context')

        # Load Detection Module
        if action == "GUI":
            print('Creating Face Detection Module in Context')
            self.detect = self.face_detector()

            print('Creating Face Recognition Module in Context')
            self.recognize = self.face_recognizer(self.configuration.recognizer_type)

            print('Creating Post Processing Module in Context')
            # Post Processing Module
            self.postprocess = postprocess.Postprocessing(self.data_loader)

    def face_detector(self):
        return detect.Detect(self.base_directory, self.configuration)

    def face_recognizer(self, type):
        return recognize.Recognize(type, self.model_loader, self.configuration, self.data_loader)


