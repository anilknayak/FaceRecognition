import numpy as np
from facerecognition.svm_facenet import preprocessing as pp
from facerecognition.svm_facenet import postprocessing as pop

class Recognize:
    def __init__(self, model_loader, depth):
        self.model_dict, self.model_svm = model_loader.load_facenet_svm_model()
        self.sess = model_loader.sess
        self.depth = depth
        self.preprocess = pp.PreProcessing()
        self.postprocess = pop.PostProcessing()

    def recognize(self, faces, no_of_pred):
        faces_preprocessed = self.preprocess.pre_process(faces)
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

        return faces

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def get_predicted_faces(self, image):
        with self.sess.as_default():
            # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            img = self.prewhiten(image)

            feed_dict = {self.model_dict['input']: [img], self.model_dict['phase_train']: False}
            self.embedding = self.sess.run(self.model_dict['embeddings'], feed_dict=feed_dict)[0]

            # print(self.embedding)
            self.predictions = self.model_svm.predict_proba([self.embedding])
            # print(self.predictions)

    def finalize_prediction(self, no_of_pred):
        self.sorted_probability_index = sorted(range(len(self.predictions[0])), key=lambda i: self.predictions[0][i])
        self.predictions_indx = self.sorted_probability_index[-no_of_pred:][::-1]

        self.pred_label = []
        self.prob_pred = []
        for pred in self.predictions_indx:
            self.pred_label.append(self.model_dict['classifier'][pred])
            self.prob_pred.append(self.predictions[0][pred])
#