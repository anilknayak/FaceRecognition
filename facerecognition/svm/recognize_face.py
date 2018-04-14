from facerecognition.svm import preprocessing as pp
from facerecognition.svm import postprocessing as pop
import numpy as np

class Recognize:
    def __init__(self, model_loader, depth):
        self.model_dict, self.model_svm = model_loader.load_svm_model(True)
        self.sess = model_loader.sess
        self.depth = depth
        self.preprocess = pp.PreProcessing()
        self.postprocess = pop.PostProcessing()

    def recognize(self, faces, no_of_pred):
        faces_preprocessed = self.preprocess.pre_process(faces, self.depth)
        faces_recognized = self.recognize_faces(faces_preprocessed, no_of_pred)
        faces_postprocessed = self.postprocess.post_process(faces_recognized)
        return faces_postprocessed

    def recognize_faces(self, faces, no_of_pred):
        frame_rate = 0
        recognized_faces = []
        for face in faces:
            face_image = face.image
            self.get_predicted_faces(face_image)
            self.finalize_prediction(no_of_pred)

            face.labels_pred = self.pred_label
            face.prob_pred = self.prob_pred

        # faces = self.face_recognition.identify(face_image)
        # self.add_overlays(image, faces, frame_rate)

        return faces

    def get_predicted_faces(self, image):
        with self.sess.as_default():
            feed_dict = {self.model_dict['input_tensor']: [image], self.model_dict['output_tensor']: -1}
            (self.embedding) = self.sess.run([self.model_dict['embedding/BiasAdd']], feed_dict=feed_dict)

            self.predictions = np.asarray(self.model_svm.predict_proba([np.asarray(np.squeeze(self.embedding[0]))]))

    def finalize_prediction(self,no_of_pred):
        self.sorted_probability_index = sorted(range(len(self.predictions[0])), key=lambda i: self.predictions[0][i])
        self.predictions_indx = self.sorted_probability_index[-no_of_pred:][::-1]

        self.pred_label = []
        self.prob_pred = []
        for pred in self.predictions_indx:
            self.pred_label.append(self.model_dict['classifier'][pred])
            self.prob_pred.append(self.predictions[0][pred])
