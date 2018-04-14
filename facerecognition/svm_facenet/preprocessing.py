import numpy as np
import cv2
from utils import image_util as iu

class PreProcessing:
    def __init__(self):
        self.image_utils = iu.ImageUtil()
        self.allowable_face_height = 160
        self.allowable_face_width = 160

    def pre_process(self, faces):
        faces_preprocessed = []
        for face in faces:
            image = face.image
            height, width , _ = np.shape(image)

            # h,w,_ = np.shape(face.container_image)
            # print(" propertion ",h//height)
            # print(w//width)

            if height > self.allowable_face_height and width > self.allowable_face_width:
                faces_preprocessed.append(face)
            else:

                if height>0 and width>0:
                    image_rescaled = cv2.resize(image, (self.allowable_face_height, self.allowable_face_width), interpolation=cv2.INTER_CUBIC)
                    face.image = image_rescaled
                    faces_preprocessed.append(face)

        return faces_preprocessed


    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y