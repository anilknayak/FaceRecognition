from facerecognition.facenet import recognize_face as facenet_recognition
from facerecognition.nn import recognize_face as nn_recognition
from facerecognition.inception_v1 import recognize_face as recognition_v1
from facerecognition.inception_v5 import recognize_face as recognition_v5
from facerecognition.svm import recognize_face as recognition_svm
from facerecognition.svm_facenet import recognize_face as recognition_svm_facenet

class Recognize:
    def __init__(self, type, model_loader, config, data):
        self.face_recognition_method = type
        self.model_loader = model_loader
        self.config = config
        self.data = data
        self.recognize_mod_facenet = None
        self.recognize_mod_nn = None
        self.nn_recognition_v1 = None
        self.nn_recognition_v5 = None
        self.nn_recognition_svm = None
        self.nn_recognition_svm_facenet = None

        self.facenet_flag = False
        self.nn_flag = False
        self.inceptionv1_flag = False
        self.inceptionv5_flag = False
        self.svm_flag = False
        self.svm_facenet_flag = False
        # self.validate_model()
        self.depth = self.config.depth

    def maintain_flag(self):
        self.facenet_flag = False
        self.nn_flag = False
        self.inceptionv1_flag = False
        self.inceptionv5_flag = False
        self.svm_flag = False
        self.svm_facenet_flag = False

    def recognize_face(self, faces, no_of_face_to_predict=1):
        if self.config.recognizer_type == "facenet":
            if not self.facenet_flag:
                self.maintain_flag()
                self.recognize_mod_facenet = facenet_recognition.Recognize()
                self.facenet_flag = True
            return self.get_face_recognition_from_facenet(faces)

        elif self.config.recognizer_type == "svm_facenet":
            if not self.svm_facenet_flag:
                self.maintain_flag()
                self.nn_recognition_svm_facenet = recognition_svm_facenet.Recognize(self.model_loader, self.depth)
                self.svm_facenet_flag = True
            return self.get_face_recognition_from_svm_facenet(faces)

        elif self.config.recognizer_type == "svm":
            if not self.svm_flag:
                self.maintain_flag()
                self.nn_recognition_svm = recognition_svm.Recognize(self.model_loader, self.depth)
                self.svm_flag = True
            return self.get_face_recognition_from_svm(faces)

        elif self.config.recognizer_type == "nn":
            if not self.nn_flag:
                self.maintain_flag()
                self.recognize_mod_nn = nn_recognition.Recognize(self.model_loader, self.depth)
                self.nn_flag = True
            return self.get_face_recognition_from_nn(faces)

        elif self.config.recognizer_type == "inception_v1":
            if not self.inceptionv1_flag:
                self.maintain_flag()
                self.nn_recognition_v1 = recognition_v1.Recognize(self.model_loader, self.depth)
                self.inceptionv1_flag = True
            return self.get_face_recognition_from_nn_v1(faces)

        elif self.config.recognizer_type == "inception_v5":
            if not self.inceptionv5_flag:
                self.maintain_flag()
                self.nn_recognition_v5 = recognition_v5.Recognize(self.model_loader, self.depth)
                self.inceptionv5_flag = True
            return self.get_face_recognition_from_nn_v5(faces)


    def get_face_recognition_from_facenet(self, faces):
        # facenet_recognition
        # self.recognize_mod = facenet_recognition.Recognize()
        faces = self.recognize_mod_facenet.recognize(faces, self.config.number_of_prediction)
        # images, titles = self.finalize_prediction()
        return faces

    def get_face_recognition_from_svm(self, faces):
        faces = self.nn_recognition_svm.recognize(faces, self.config.number_of_prediction)
        return faces

    def get_face_recognition_from_nn(self, faces):
        faces = self.recognize_mod_nn.recognize(faces, self.config.number_of_prediction)
        return faces

    def get_face_recognition_from_nn_v1(self, faces):
        faces = self.nn_recognition_v1.recognize(faces, self.config.number_of_prediction)
        return faces

    def get_face_recognition_from_nn_v5(self, faces):
        faces = self.nn_recognition_v5.recognize(faces, self.config.number_of_prediction)
        return faces

    def get_face_recognition_from_svm_facenet(self, faces):
        faces = self.nn_recognition_svm_facenet.recognize(faces, self.config.number_of_prediction)
        return faces

    # def validate_model(self):
    #     _, image = self.data.get_class_label_name_and_label_image('X1')
    #     f = face.Face()
    #     f.container_image = image
    #     f.image = image
    #     faces = [f]
    #
    #     try:
    #         print('facenet testing start')
    #         self.get_face_recognition_from_facenet(faces)
    #     except:
    #         print('error Loading recognition models facenet')
    #
    #     try:
    #         self.get_face_recognition_from_nn(faces)
    #     except:
    #         print('error Loading recognition models nn')
    #
    #     try:
    #         self.get_face_recognition_from_nn_v1(faces)
    #     except:
    #         print('error Loading recognition models v1')
    #
    #     try:
    #         self.get_face_recognition_from_nn_v5(faces)
    #     except:
    #         print('error Loading recognition models v5')
    #
    #     try:
    #         self.get_face_recognition_from_svm(faces)
    #     except:
    #         print('error Loading recognition models svm')
