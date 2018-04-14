import os
import pickle
import cv2
from utils import data
import datetime
import os
import platform
import numpy as np


class DataLoader:
    def __init__(self, basedir, config):
        self.configuration = config
        self.base_dir = basedir
        self.raw_images = None
        self.images = None
        self.labels = None
        self.camera = None
        self.data = None
        self.width = None
        self.height = None

    def load_training_data(self):
        if self.configuration.prepare_data_from_pickle_file:
            print("Loading training data from saved pickle file")
            self.load_pickle_file_data()
        else:
            if self.configuration.pre_processing_required:
                print("Raw Data before pre processing for training")
                self.load_training_images(self.configuration.raw_data_images, self.configuration.raw_data_labels)
            else:
                print("Loading already processed data for training")
                self.load_training_images(self.configuration.processed_data_images, self.configuration.processed_data_labels)

        return self.data

    def load_training_images(self, image_dir, label_dir):
        print("Loading Images and Labels  from dir", image_dir, label_dir)
        classes, classes_count = self.find_classes_for_training(image_dir)
        if classes is not None:
            self.data = data.Data()
            self.data.classes = classes
            self.data.classes_count = classes_count

            self.load_classes_images_for_training(image_dir, self.data.classes)
            self.load_classes_labels_for_training(label_dir, self.data.classes)

    def load_classes_images_for_training(self, image_dir, classes):

        class_label_wise_images = {}
        path = os.path.join(self.base_dir, image_dir)
        print("Loading Images for training",path)
        dict_raw_image_stat = {}
        for class_label in classes:
            path_class_images = os.path.join(path, class_label)
            images_per_labels = os.listdir(path_class_images)
            images = []
            for image in images_per_labels:
                image_path = os.path.join(path_class_images, image)
                image_file = cv2.imread(image_path)

                if image_file is not None:
                    h,w,d = np.shape(image_file)
                    if h >0 and w>0 and d>0:
                        images.append(image_file)

            class_label_wise_images[class_label] = images
            dict_raw_image_stat[class_label] = len(images)
            print("Image Label", class_label, "Number of Images",str(len(images)))

        if self.data is not None:
            self.data.class_label_wise_images = class_label_wise_images
            self.data.data_as_whole = self.data.class_label_wise_images
            self.data.dict_raw_data_stat = dict_raw_image_stat

    def load_classes_labels_for_training(self, label_dir, classes):
        print("Loading Label Images for training",label_dir)
        class_label_images = {}
        for class_label in classes:
            path_class_images = os.path.join(label_dir, class_label)
            image_file = cv2.imread(path_class_images)
            class_label_images[class_label] = image_file

        if self.data is not None:
            self.data.class_label_images = class_label_images

    def find_classes_for_training(self, image_dir):
        classes = None
        classes_count = None
        path = os.path.join(self.base_dir, image_dir)
        files = os.listdir(path)
        count = 1
        for dirs in files:
            if os.path.isdir(os.path.join(path, dirs)):
                if count == 1:
                    classes = []
                    classes_count = []
                classes.append(dirs)
                classes_count.append(count)
                count = count + 1

        return classes, classes_count

    def load_pickle_file_data(self):
        ''

    def load_label_image(self):
        label_data_path = os.path.join(self.base_dir, "data/training_label.pickle")
        label_pickle = open(label_data_path, "rb")
        label_pickle_dict = pickle.load(label_pickle)
        self.classes = label_pickle_dict['classes']
        self.classes_number = label_pickle_dict['classes_count']
        self.class_label_images = label_pickle_dict['label_images']
        self.labels_number_mapping = label_pickle_dict['label_int_mapping']


    def get_raw_training_images(self):
        return self.raw_images

    def get_image_labels(self):
        return self.labels

    def get_training_images(self):
        return self.images

    def get_data(self):
        return self.images, self.labels

    def get_class_label_name_and_label_image(self, prediction):
        label_name = prediction
        label_data_path = os.path.join(self.base_dir, "data/labels/"+label_name+".jpg")
        label_image = cv2.imread(label_data_path)
        # label_image = self.class_label_images[prediction]
        # print(np.shape(label_image))
        return label_name, label_image, label_data_path

    def get_label_names(self, indexs):
        # labels = []
        # for index in indexs:
        #     label = self.classes[index]
        #     labels.append(label)
        # return labels

        labels = []
        for index in indexs:
            label = self.labels_number_mapping[index]
            labels.append(label)
        return labels

    def initialize_camera(self, camera_no):

        print()

        self.camera = cv2.VideoCapture(camera_no)

        # self.fourcc = self.camera.VideoWriter_fourcc(*'X264')

    def video_writter(self, dir, codec):
        uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':','-')
        split_uniq_filename = os.path.join(dir, uniq_filename + ".avi")
        self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # self.fourcc = cv2.VideoWriter_fourcc(*codec)

        if codec == 'X264':
            self.fourcc = cv2.VideoWriter_fourcc(*'X264')
        elif codec == 'XVID':
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # if platform.system() == 'Darwin':
        #
        # elif platform.system() == 'Linux':
        #

        self.writter = cv2.VideoWriter(split_uniq_filename, self.fourcc, 30.0, (int(self.width), int(self.height)))

        return self.writter

    def destroy_camera(self):
        self.camera.release()
        cv2.destroyAllWindows()

    def get_frame_from_camera(self):
        has_frame, frame = self.camera.read()

        if has_frame:
            return frame

        return None


