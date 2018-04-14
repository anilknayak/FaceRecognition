import cv2
import datetime
import os
from collections import Counter
import numpy as np

class Postprocessing:
    def __init__(self, data):
        self.data = data
        self.accumulator = []
        self.accumulator_weight = []
        self.weights = [1.0, 0.5, 0.2, 0.1, 0.01]
        self.started = False
        self.change_in_accumulator = False

    def draw_rectangle(self, faces, frame):
        for face_arr in faces:
            x = face_arr[1][0]
            y = face_arr[1][1]
            x_w = face_arr[1][2]
            y_h = face_arr[1][3]
            cv2.rectangle(frame, (x, y), (x_w, y_h), (0, 255, 0), 2)
        return frame

    def add_overlays(self, frame, faces, type, show_feature_points):
        if type == 'All Faces':
            if faces is not None:
                for face in faces:
                    face_bb = face.bounding_box.astype(int)
                    cv2.rectangle(frame,
                                  (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                                  (0, 255, 0), 2)

                    if show_feature_points and face.features is not None:
                        for pt in face.features:
                            cv2.circle(frame, (pt[0], pt[1]), radius=5, color=(0, 255, 0))

        elif type == 'Near to Camera':
            max_area = 0
            for face in faces:
                if max_area < face.area:
                    max_area = face.area

            for face in faces:
                if face.area == max_area:
                    face_bb = face.bounding_box
                    cv2.rectangle(frame,
                                  (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                                  (0, 255, 0), 2)

                    if show_feature_points and face.features is not None:
                        for pt in face.features:
                            cv2.circle(frame, (pt[0], pt[1]), radius=5, color=(0, 255, 0))

        return frame

    def reset_accumulator(self):
        print('accumulator reset complete')
        self.accumulator = []
        self.started = False

    def postprocessing_face_pred(self, faces, config):
        faces_accumulated = []

        # try:
        if config.accumulator_status and config.number_of_recognized==1:
            for face in faces:
                labels = np.copy(face.labels_pred)
                probs = np.copy(face.prob_pred)
                face.recognition_stat = []

                if config.accumulator_weighted_status:
                    # accumulator for weighted average of all the prediction first prediction
                    if len(self.accumulator) > (config.accumulator_frame_count * config.number_of_prediction):
                        reduceby = len(self.accumulator) - (config.accumulator_frame_count * config.number_of_prediction)
                        self.accumulator = self.accumulator[reduceby:]
                        self.accumulator_weight = self.accumulator_weight[reduceby:]

                    if len(self.accumulator) == (config.accumulator_frame_count * config.number_of_prediction):
                        total = len(self.accumulator)
                        dict = {}
                        dict_count = {}
                        # calculate the weighted accumulator
                        for i in range(len(self.accumulator)):
                            w = self.accumulator_weight[i]
                            x = self.accumulator[i]

                            if x in dict.keys():
                                dict[x] = float(dict[x]) + float(w)
                                dict_count[x] = int(dict_count[x]) + 1
                            else:
                                dict[x] = float(w)
                                dict_count[x] = 1

                        sorted_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

                        # print(sorted_dict)
                        # print(dict_count)
                        # print("====================")

                        if len(sorted_dict) >= config.number_of_prediction:
                            cnt = 0
                            for k, v in sorted_dict:
                                face.labels_pred[cnt] = k
                                correct = dict_count[k]
                                face.recognition_stat.append(str(str(correct) + " / " + str(total)))
                                cnt = cnt+1
                                if cnt == config.number_of_prediction:
                                    break
                        else:
                            cnt = 0
                            max_len = len(sorted_dict)
                            for k, v in sorted_dict:
                                face.labels_pred[cnt] = k
                                correct = dict_count[k]
                                face.recognition_stat.append(str(str(correct) + " / " + str(total)))
                                cnt = cnt+1
                                if cnt == max_len:
                                    break

                        # print(len(self.accumulator), self.accumulator)
                        # print(len(self.accumulator_weight), self.accumulator_weight)

                        dict = {}
                        dict_count = {}
                        face.images_pred = []
                        face.image_paths = []

                        if face.labels_pred is not None:
                            for name in face.labels_pred:
                                _, label_image, label_data_path = self.data.get_class_label_name_and_label_image(name)
                                face.images_pred.append(label_image)
                                face.image_paths.append(label_data_path)


                        self.accumulator = self.accumulator[config.number_of_prediction:]
                        self.accumulator_weight = self.accumulator_weight[config.number_of_prediction:]

                        for i in range(config.number_of_prediction):
                            prob = probs[i]
                            label = labels[i]
                            self.accumulator.append(label)
                            self.accumulator_weight.append(prob)
                        self.started = True
                    else:
                        for i in range(config.number_of_prediction):
                            prob = probs[i]
                            label = labels[i]
                            self.accumulator.append(label)
                            self.accumulator_weight.append(prob)
                        self.started = False

                # if len(self.accumulator) == (config.accumulator_frame_count * config.number_of_prediction) and self.started:
                #     faces_accumulated = faces

                else:
                    # accumulator for not weighted predictions
                    if len(self.accumulator) > (config.accumulator_frame_count * config.number_of_prediction):
                        reduceby = len(self.accumulator) - (config.accumulator_frame_count * config.number_of_prediction)
                        self.accumulator = self.accumulator[reduceby:]
                        self.accumulator_weight = self.accumulator_weight[reduceby:]

                    if len(self.accumulator) == (config.accumulator_frame_count * config.number_of_prediction):
                        total = len(self.accumulator)

                        most_common = Counter(self.accumulator).most_common(config.number_of_prediction)  # X1, 6 times

                        if len(most_common) >= config.number_of_prediction:
                            cnt = 0
                            for x, c in most_common:
                                face.labels_pred[cnt] = x
                                correct = c
                                face.recognition_stat.append(str(str(correct) + " / " + str(total)))
                                cnt = cnt + 1
                                if cnt == config.number_of_prediction:
                                    break
                        else:
                            cnt = 0
                            max_len = len(most_common)
                            for x, c in most_common:
                                face.labels_pred[cnt] = x
                                correct = c
                                face.recognition_stat.append(str(str(correct) + " / " + str(total)))
                                cnt = cnt + 1
                                if cnt == max_len:
                                    break

                        # print(len(self.accumulator),self.accumulator)
                        # print(len(self.accumulator_weight), self.accumulator_weight)
                        # print(len(most_common), most_common)

                        face.images_pred = []
                        face.image_paths = []

                        if face.labels_pred is not None:
                            for name in face.labels_pred:
                                _, label_image, label_data_path = self.data.get_class_label_name_and_label_image(name)
                                face.images_pred.append(label_image)
                                face.image_paths.append(label_data_path)

                        self.accumulator = self.accumulator[config.number_of_prediction:]
                        self.accumulator_weight = self.accumulator_weight[config.number_of_prediction:]

                        for i in range(config.number_of_prediction):
                            prob = probs[i]
                            label = labels[i]
                            self.accumulator.append(label)
                            self.accumulator_weight.append(prob)
                        self.started = True
                    else:
                        for i in range(config.number_of_prediction):
                            prob = probs[i]
                            label = labels[i]
                            self.accumulator.append(label)
                            self.accumulator_weight.append(prob)
                        self.started = False

            if len(self.accumulator) == (
                config.accumulator_frame_count * config.number_of_prediction) and self.started:
                faces_accumulated = faces
        else:
            self.accumulator = []
            self.accumulator_weight = []
            if faces is not None:
                for face in faces:
                    face.images_pred = []
                    face.image_paths = []

                    if face.labels_pred is not None:
                        for name in face.labels_pred:
                            _, label_image, label_data_path = self.data.get_class_label_name_and_label_image(name)
                            face.images_pred.append(label_image)
                            face.image_paths.append(label_data_path)

            faces_accumulated = faces
        # except:
        #     print('error')

        return faces_accumulated

    def calculate_face_area(self,faces):
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            x = face_bb[0]
            y = face_bb[1]
            w = face_bb[2]
            h = face_bb[3]
            area = abs((int(w) - int(x)) * (int(h) - int(y)))
            face.area = area

        return faces

    def save_faces(self, frame, faces, dir, type):
        max_area = 0
        if type == 'Near to Camera':
            for face in faces:
                if max_area < face.area:
                    max_area = face.area
            for face in faces:
                if max_area == face.area:
                    self.write_face(face, frame,dir)
        elif type == 'All Faces':
            for face in faces:
                self.write_face(face,frame,dir)

    def write_face(self, face, frame, dir):
        uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':','-')
        split_uniq_filename = os.path.join(dir, uniq_filename + ".jpg")
        face_bb = face.bounding_box #.astype(int)
        x = face_bb[0]
        y = face_bb[1]
        w = face_bb[2]
        h = face_bb[3]
        face = frame[y: h, x: w, :]
        cv2.imwrite(split_uniq_filename, face)