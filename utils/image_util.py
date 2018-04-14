import cv2
import numpy as np

class ImageUtil:
    def __init__(self):
        ''

    def normalize_face(self, face):
        gray_scale_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        image_between_0_and_1 = gray_scale_image / 255.0
        image_between_0_and_1 = image_between_0_and_1 - 0.5
        normalized_image_between_ng_1_and_po_1 = image_between_0_and_1 * 2.0
        return normalized_image_between_ng_1_and_po_1

    def reshape(self, face, height, width):
        reshaped_face = face.reshape((height, width, 1))
        return reshaped_face

    def resize(self, face, height, width):
        resized_face = cv2.resize(face, (height, width), interpolation=cv2.INTER_CUBIC)
        return resized_face


