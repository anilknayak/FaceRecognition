from tqdm import tqdm
import cv2
import numpy as np
from utils import image_util as iu

class Preprocessing:
    def __init__(self, configuration):
        self.configuration = configuration
        self.image_utils = iu.ImageUtil()

    def preprocessing_training(self, datas, action, depth=1):
        if depth == 1:
            return self.preprocessing_training_1(datas, action)
        elif depth == 3:
            return self.preprocessing_training_2(datas)

    def preprocessing_training_2(self, datas):
        images_label_wise_dict = datas.class_label_wise_images
        faces_per_label_normalized = {}
        for label_images in tqdm(images_label_wise_dict, desc="Pre Processing Label Images"):
            images = images_label_wise_dict[label_images]
            faces_normalized = []
            for image in images:
                img = image
                if image.ndim == 2:
                    img = self.to_rgb(img)
                img = self.normalize_face(img)
                faces_normalized.append(img)
            faces_per_label_normalized[label_images] = faces_normalized

        datas.data_as_whole = faces_per_label_normalized

        return datas

    def normalize_face(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    def preprocessing_training_1(self, datas, actions):
        images_label_wise_dict = datas.class_label_wise_images
        pre_proc_data = None

        if 'fetchface' in actions:
            faces_per_label = {}
            for label_images in tqdm(images_label_wise_dict, desc="For label finding faces"):
                images = images_label_wise_dict[label_images]
                face_images = []
                for image in tqdm(images, desc="finding faces in image"):
                    faces = self.face_capture(image)
                    for face in faces:
                        face_images.append(face)

                faces_per_label[label_images] = face_images

            pre_proc_data = faces_per_label
        else:
            pre_proc_data = images_label_wise_dict


        if 'normalize' in actions:
            faces_per_label_normalized = {}
            for label in tqdm(pre_proc_data, desc="Normalizing Images"):
                images = pre_proc_data[label]
                faces_normalized = []
                for image in images:
                    face = self.image_utils.normalize_face(image)
                    faces_normalized.append(face)

                faces_per_label_normalized[label] = faces_normalized

            pre_proc_data = faces_per_label_normalized

        if 'resize' in actions:
            faces_per_label_resize = {}
            for label in tqdm(pre_proc_data, desc="Resizing Images"):
                images = pre_proc_data[label]
                faces_resize = []
                for image in images:
                    face = self.image_utils.resize(image, self.configuration.height, self.configuration.width)
                    faces_resize.append(face)

                faces_per_label_resize[label] = faces_resize

            pre_proc_data = faces_per_label_resize

        if 'reshape' in actions:
            faces_per_label_reshape = {}
            for label in tqdm(pre_proc_data, desc="Reshaping Images"):
                images = pre_proc_data[label]
                faces_reshaped = []
                for image in images:
                    face = self.image_utils.reshape(image, self.configuration.height, self.configuration.width)
                    faces_reshaped.append(face)

                faces_per_label_reshape[label] = faces_reshaped

            pre_proc_data = faces_per_label_reshape

        datas.data_as_whole = pre_proc_data

        return datas

    def face_capture(self, image):
        ''

    # def preprocess_data(self, datas, actions):
    #     for data in datas:
    #         image = data.image
    #         for action in actions:
    #             if action == 'normalize':
    #                 image = self.normalize_face(image)
    #             if action == 'reshape':
    #                 image = self.reshape(image)
    #
    #         data.image = image
    #
    #     return datas

    # def normalize_face(self, face):
    #     face = cv2.resize(face, (self.configuration.height, self.configuration.width), interpolation=cv2.INTER_CUBIC)
    #     gray_scale_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    #     image_between_0_and_1 = gray_scale_image / 255.0
    #     image_between_0_and_1 = image_between_0_and_1 - 0.5
    #     normalized_image_between_ng_1_and_po_1 = image_between_0_and_1 * 2.0
    #     return normalized_image_between_ng_1_and_po_1
    #
    # def reshape(self, face):
    #     reshaped_face = face.reshape((1, self.configuration.height, self.configuration.width, 1))
    #     return reshaped_face
    #
    # def resize(self, face):
    #     resized_face = cv2.resize(face, (self.configuration.height, self.configuration.width), interpolation=cv2.INTER_CUBIC)
    #     return resized_face