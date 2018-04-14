import dlib
from imutils import face_utils
import os

class FindFace:
    def __init__(self, base_directory):
        self.detector = dlib.get_frontal_face_detector()
        path = os.path.join(base_directory, "facedetection/dlib/dlib_pretrained_model.dat")
        self.predictor = dlib.shape_predictor(path)

    def getfaces(self, image):
        rects = self.detector(image, 1)
        boxes = []
        try:
            for rect in rects:
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                box = [x, y, x + w, y + h]
                # box = factor * box
                # box1 = [box[0] - 10, box[1] - 50, box[0] + box[2] + 20, box[1] + box[3] + 10, 0]
                # boxes.append(box1)
                boxes.append(box)
        except:
            ''
        return boxes