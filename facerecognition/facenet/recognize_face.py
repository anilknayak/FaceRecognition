import numpy as np
from facerecognition.facenet import face
import cv2

class Recognize:
    def __init__(self):
        self.face_recognition = face.Recognition()
        self.allowable_face_height = 160
        self.allowable_face_width = 160



    def recognize(self, faces, no_of_pred):
        faces_preprocessed = self.pre_process(faces)
        faces = self.face_recognition.recognize_face(faces_preprocessed, no_of_pred)
        return faces

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
                    image_rescaled = cv2.resize(image,(self.allowable_face_height, self.allowable_face_width), interpolation=cv2.INTER_CUBIC)
                    face.image = image_rescaled
                    faces_preprocessed.append(face)

        return faces_preprocessed

