import numpy as np
import cv2
from utils import image_util as iu

class PreProcessing:
    def __init__(self):
        self.image_utils = iu.ImageUtil()

    def pre_process(self, datas, depth):
        if depth == 3:
            return self.pre_process_depth(datas)
        elif depth == 1:
            return self.pre_process_1(datas)

    def pre_process_1(self, datas):
        actions = ['normalize', 'resize', 'reshape']
        height = 120
        width = 120

        faces = []

        for data in datas:
            image = data.image

            h,w,d = np.shape(image)

            if h>0 and w>0:
                for action in actions:
                    if action == 'normalize':
                        image = self.image_utils.normalize_face(image)
                    elif action == 'reshape':
                        image = self.image_utils.reshape(image, height, width)
                    elif action == 'resize':
                        image = self.image_utils.resize(image, height, width)
                data.image = image
                faces.append(data)

        return faces

    def pre_process_depth(self, datas):
        height = 120
        width = 120

        faces = []

        for data in datas:
            image = data.image

            h, w, d = np.shape(image)

            if h > 0 and w > 0:
                if image.ndim == 2:
                    image = self.to_rgb(image)
                image = self.normalize_face1(image)
                image = cv2.resize(image, (height, width), interpolation=cv2.INTER_CUBIC)
                data.image = image
                faces.append(data)

        return faces

    def normalize_face1(self, x):
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