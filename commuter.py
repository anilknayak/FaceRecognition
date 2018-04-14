import context


class Commuter:
    def __init__(self, action, debug_flag=False):
        self.context = context.Context(action, debug_flag)

    def load_configuration(self, configuration_file):
        ''

    def load_model_embedding(self, configuration_file, clear):
        model = self.context.model_loader.load_inception_v5_model(clear)
        return model

    def load_facenet_embedding(self, configuration_file, clear):
        model = self.context.model_loader.load_facenet_model(clear)
        return model

    def reset_accumulator(self):
        self.context.postprocess.reset_accumulator()

    def load_data(self):
        # This method will return data as follows
        """
            This returns a Data Object
            class_label_wise_images: is a dictionary
                Key: Label Name
                Value: List of Images
            class_label_images: is a dictionary
                Key: Label Name
                Value: Label Image
            classes: is a List
                Each value is a Label Name

            :return: Data Object
        """

        return self.context.data_loader.load_training_data()

    def pre_processing(self, datas, actions, type, depth):
        """
            This returns a Data Object
            class_label_wise_images: is a dictionary
                Key: Label Name
                Value: List of Images
            class_label_images: is a dictionary
                Key: Label Name
                Value: Label Image
            classes: is a List
                Each value is a Label Name

            data_as_whole: is a dictionary
                Key: Label
                Value: List of Images

            :return: Data Object
        """
        if type == 'GUI':
            return datas
        elif type == 'TRAIN':
            return self.context.preprocess.preprocessing_training(datas, actions, depth)
        # elif type == 'PROCESS':
        #     return self.context.preprocess.preprocess_data(datas, actions)

    def data_augmentation(self, data, flag=True, depth=1, reshape=True):
        '''
            data: is a data object
            data.classes: List of Class Labels
            data.data_as_whole: is a dictionary Key: Label and Value: List of Images
            :param: flag if true then augment or if false then just rearrange the data as per return values
            :return: data object having images and labels,
                     labels_number_mapping is a dictionary having keys as counter and value as Labels in Str format
        '''
        temp_data = self.context.augment.augment(data, flag, depth, reshape)
        # self.save_data_labels_in_pickle(temp_data)
        return temp_data

    # def save_data_labels_in_pickle(self, data):
    #     self.context.data_loader.save_pickle_file_for_training_labels(data)

    def data_separation(self, data, action='NORMAL'):
        """


        :param data: Object having Images and Labels
        :param action:
        :return:
            data.training_data = training dictionary having key as images and labels_n
            data.testing_data = testing dictionary having key as images and labels_n
            data.validation_data = validation dictionary having key as images and labels_n
        """
        if action == 'NORMAL':
            data1 = self.context.data_separation.separate_data(data)
        elif action == 'CROSS_VALIDATION':
            data1 = None

        return data1

    def prepare_deep_network(self, path, depth=1):
        self.context.model_preparation.load_nn_architecture(path)
        self.context.model_preparation.build_nn_model(depth)

        return self.context.model_preparation

    def prepare_batch(self, images, labels, batch_size):
        return self.context.batch_prepare.prepare(images, labels, batch_size)

    def training(self, model_obj, data_obj):
        ''

    def testing(self):
        ''

    def validation(self):
        ''

    def model_freeze(self):
        ''

    def initialize_camera(self, source, video_file):
        if source == 'Camera':
            print('camera initialized', self.context.configuration.camera)
            self.context.data_loader.initialize_camera(self.context.configuration.camera)
        elif source == 'Video File':
            print('initializing video file',video_file)
            self.context.data_loader.initialize_camera(video_file)

    def stop_camera(self):
        print('in side commuter stop camera')
        self.context.data_loader.destroy_camera()
        self.context.configuration.start_camera = False

    def get_video_capture_obj(self, capture_dir, codec):
        return self.context.data_loader.video_writter(capture_dir, codec)

    def get_input(self, type):
        input = None
        # get input from camera or batch processing details
        if type == 'camera':
            input = self.context.data_loader.get_frame_from_camera()
        elif type == 'image':
            input = None #image
        elif type == 'data':
            input = None #batch

        return input

    def detect(self, frame):
        # List of Face Object
        faces = self.context.detect.get_faces(frame)
        return faces

    def recognize(self, faces):
        faces = self.context.recognize.recognize_face(faces, self.context.configuration.number_of_prediction)
        return faces

    def post_processing(self, faces, config):
        faces = self.context.postprocess.postprocessing_face_pred(faces, config)
        return faces

    def faces_area_calculation(self, faces):
        return self.context.postprocess.calculate_face_area(faces)

    def save_faces(self, frame, faces, dir, type):
        self.context.postprocess.save_faces(frame, faces, dir, type)

    def display(self, faces, frame, type, show_feature_points):
        processed_frame = self.context.postprocess.add_overlays(frame, faces, type, show_feature_points)
        return processed_frame

    # This method will be used for GUI to change the detector
    def change_context_fece_detector(self, type):
        self.context.detect.face_detection_method = type

    # This method will be used for the GUI to change the recognizer
    def change_context_fece_recognizer(self, type):
        self.context.recognize.face_recognition_method = type

    # This method will be used for the GUI to change the recognizer
    def change_context_no_of_prediction(self, no_of_pred):
        self.context.configuration.number_of_prediction = no_of_pred

    def get_no_of_prediction(self):
        return self.context.configuration.number_of_prediction

    # This method will be used for the GUI to change the recognizer
    def change_context_camera(self, camera):
        self.context.configuration.start_camera = False
        self.context.configuration.camera = camera
        self.context.configuration.start_camera = True

    def get_camera(self):
        return self.context.configuration.camera

    def start_camera(self):
        self.context.configuration.start_camera = True









