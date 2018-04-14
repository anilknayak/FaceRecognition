
from facerecognition.nn import preprocessing as pp
from facerecognition.nn import postprocessing as pop

class Recognize:
    def __init__(self, model_loader, depth):
        self.model_dict = model_loader.load_nn_cnn_model(True)
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
            (self.output,self.prob) = self.sess.run([self.model_dict['output/BiasAdd'],
                                                     self.model_dict['predicted_class_probabilities']],
                                                        feed_dict={self.model_dict['input_tensor']: [image], self.model_dict['output_tensor']: -1})

    def finalize_prediction(self,no_of_pred):
        self.sorted_probability_index = sorted(range(len(self.prob[0])), key=lambda i: self.prob[0][i])
        self.predictions = self.sorted_probability_index[-no_of_pred:][::-1]

        self.pred_label = []
        self.prob_pred = []
        for pred in self.predictions:
            self.pred_label.append(self.model_dict['classifier'][pred])
            self.prob_pred.append(self.prob[0][pred])
