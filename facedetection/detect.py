import cv2
import numpy as np
from utils import face as faceobj
from facedetection.dlib import find_face as dlibff
from facedetection.mtcnn import find_face as mtcnnff
import tensorflow as tf
import math


class Detect:
    def __init__(self, base_dir, config):
        self.base_directory = base_dir
        self.config = config
        self.face_detection_method = "mtcnn"
        self.mtcnn_obj = mtcnnff.FindFace(self.base_directory)
        self.dlib_obj = dlibff.FindFace(self.base_directory)

        self.allowable_face_height_min = 90
        self.allowable_face_width_min = 90

        self.allowable_face_height_max = 800
        self.allowable_face_width_max = 800

    def get_faces(self, image):
        original_image = np.copy(image)

        downsample_image = self.pre_process(image, self.config.down_sampling_factor)
        boxes, points = self.detect_face(downsample_image)
        faces = self.post_process(original_image, boxes, points, self.config.down_sampling_factor)

        return faces

    def get_without_process_faces(self, image):
        boxes = self.detect_face(image)
        faces = []
        for box in boxes:
            box_int = np.int32(np.round(box))
            face = faceobj.Face()
            face.bounding_box = box_int
            face.image = image[face.bounding_box[1]:face.bounding_box[3],
                         face.bounding_box[0]:face.bounding_box[2], :]

            h, w, d = np.shape(face.image)
            if h > 0 and w > 0 and d > 0:
                faces.append(face)

        return faces

    def pre_process(self, image, factor):
        downsampled_image = image[::factor, ::factor, :]
        return downsampled_image

    def detect_face(self, image):
        if self.config.detector_type == "mtcnn":
            return self.mtcnn_obj.getfaces(image)
        elif self.config.detector_type == "dlib":
            return self.dlib_obj.getfaces(image), None

    def post_process(self, original_image, boxes, points, factor):
        if points is not None:
            points = points * factor

        faces = []
        number_of_boxes = len(boxes)

        for index in range(number_of_boxes):
            box = boxes[index]

            # print("before multiply factor ", self.config.detector_type, " : ", box)

            box_factor = np.asarray(box) * factor
            box_factor[0:4] = np.asarray(box_factor[0:4]) + np.asarray([  -1 * int(self.config.boundingbox_size_incr_by),
                                                                          -1 * int(self.config.boundingbox_size_incr_by),
                                                                          int(self.config.boundingbox_size_incr_by),
                                                                          int(self.config.boundingbox_size_incr_by)
                                                                       ])

            box_int = np.int32(np.round(box_factor))
            face = faceobj.Face()

            # print("after multiply factor ", self.config.detector_type , " : ", box_int)

            if (self.config.show_feature_points or self.config.vertical_align_face) and points is not None:
                point_for_box_x = points[0:5]
                point_for_box_y = points[5:]
                feature = []
                for i in range(5):
                    x = point_for_box_x[i][index]
                    y = point_for_box_y[i][index]
                    feature.append([x, y])

                face.features = feature

            face.container_image = original_image
            face.bounding_box = box_int

            big_x = 0
            big_y = 0
            big_h = 0
            big_w = 0

            o_h, o_w, o_d = np.shape(original_image)

            if face.bounding_box[0] > self.config.add_padding:
                big_x = face.bounding_box[0] - self.config.add_padding

            if face.bounding_box[2] < o_w - self.config.add_padding:
                big_w = face.bounding_box[2] + self.config.add_padding

            if face.bounding_box[1] > self.config.add_padding:
                big_y = face.bounding_box[1] - self.config.add_padding

            if face.bounding_box[3] < o_h - self.config.add_padding:
                big_h = face.bounding_box[3] + self.config.add_padding

            if big_x > 0 and big_y > 0 and big_h > 0 and big_w > 0:
                face.big_face_image = original_image[big_y:big_h, big_x:big_w, :]
            else:
                face.big_face_image = None

            face.image = original_image[face.bounding_box[1]:face.bounding_box[3],
                         face.bounding_box[0]:face.bounding_box[2], :]
            face.image_rgb = face.image
            # print("face.image", np.shape(face.image))
            faces.append(face)

        faces = self.limit_range_of_proportion(faces)
        faces = self.calculate_face_area(faces)
        faces = self.how_many_face_has_to_be_detected(faces)

        if self.config.vertical_align_face and points is not None:

            faces = self.make_face_straight(faces)

        return faces

    def make_face_straight(self, faces):
        for face in faces:
            feature = face.features
            image = face.image
            big_face_image = face.big_face_image
            lefteye = feature[0]
            righteye = feature[1]
            ang = self.angle(lefteye, righteye)

            rows_s, cols_s, _ = image.shape

            if big_face_image is not None and not self.config.add_padding == 0:
                rows_b, cols_b, _ = big_face_image.shape
                if rows_b > 0 and cols_b > 0:
                    M = cv2.getRotationMatrix2D((cols_b / 2, rows_b / 2), (-1) * ang, 1)
                    dst = cv2.warpAffine(big_face_image, M, (cols_b, rows_b))
                    dst_h, dst_w, dst_d = np.shape(dst)
                    x = 0
                    x1 = 0
                    y = 0
                    y1 = 0

                    if dst_w > (self.config.add_padding * 2) and dst_h > (self.config.add_padding * 2):
                        x = self.config.add_padding
                        x1 = dst_w - self.config.add_padding
                        y = self.config.add_padding
                        y1 = dst_h - self.config.add_padding
                        crop_img = dst[y:y1, x:x1, :]
                        face.image = crop_img

                elif rows_s > 0 and cols_s > 0:
                    M = cv2.getRotationMatrix2D((cols_s / 2, rows_s / 2), (-1) * ang, 1)
                    dst = cv2.warpAffine(image, M, (cols_s, rows_s))
                    face.image = dst
            else:
                if rows_s > 0 and cols_s > 0:
                    M = cv2.getRotationMatrix2D((cols_s / 2, rows_s / 2), (-1) * ang, 1)
                    dst = cv2.warpAffine(image, M, (cols_s, rows_s))
                    face.image = dst

        return faces

    def angle(self, pt1, pt2):
        m1 = (pt1[1] - pt1[1]) / 1
        m2 = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        tnAngle = (m1 - m2) / (1 + (m1 * m2))
        return math.atan(tnAngle) * 180 / math.pi

    def limit_range_of_proportion(self, faces):
        face_selected = []
        if faces is not None:
            for face in faces:
                box = face.bounding_box.astype(int)
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                if ((w > self.allowable_face_width_min and h > self.allowable_face_height_min)
                    or (w < self.allowable_face_width_max and h < self.allowable_face_height_max)):
                    face_selected.append(face)

        return face_selected

    def how_many_face_has_to_be_detected(self, faces):
        if not self.config.number_of_recognized == 'all':
            no_of_face = int(self.config.number_of_recognized)
            if self.config.debug:
                print("returning " + str(
                    self.config.number_of_recognized) + " number of faces for recognition from detector")
            return faces[:no_of_face]
        else:
            if self.config.debug:
                print("returning all the faces for recognition from detector")
            return faces

    def calculate_face_area(self, faces):
        for face in faces:
            face_bb = face.bounding_box.astype(int)

            x = face_bb[0]
            y = face_bb[1]
            w = face_bb[2]
            h = face_bb[3]

            area = abs((int(w) - int(x)) * (int(h) - int(y)))
            face.area = area

            # if w > 90 and h > 90:
            #     area = abs((int(w) - int(x)) * (int(h) - int(y)))
            #     face.area = area
            # else:
            #     face.area = 0

        faces.sort(key=lambda face: face.area, reverse=True)

        return faces

        # def old(self, frame, boxes, image):
        #     faces = []
        #     for box in boxes:
        #         face = faceobj.Face()
        #         face.container_image = image
        #
        #         box[0:2] -= 32
        #         box[2:] += 32
        #         box[box < 0] = 0
        #         box = np.int32(np.round(box))
        #         f = image[box[1]:box[3], box[0]:box[2]]
        #         max_dim = max(f.shape)
        #         f = cv2.resize(f, (int(f.shape[1] / max_dim * 224), int(f.shape[0] / max_dim * 224)))
        #         face.bounding_box[0] = box[0]
        #         face.bounding_box[1] = box[1]
        #         face.bounding_box[2] = box[2]
        #         face.bounding_box[3] = box[3]
        #         face.image = frame[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        #
        #         faces.append(face)
        #
        #
        #     return faces

        # def normalize_face(self, faces):
        #     faces_normalized = []
        #     for face_arr in faces:
        #         face = face_arr[0]
        #         face_cord = face_arr[1]
        #         face = cv2.resize(face, (180, 180), interpolation=cv2.INTER_CUBIC)
        #         gray_scale_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #         image_between_0_and_1 = gray_scale_image / 255.0
        #         image_between_0_and_1 = image_between_0_and_1 - 0.5
        #         normalized_image_between_ng_1_and_po_1 = image_between_0_and_1 * 2.0
        #         frame1 = normalized_image_between_ng_1_and_po_1.reshape((1, 180, 180, 1))
        #         faceArr = []
        #         faceArr.append(frame1)
        #         faceArr.append(face_cord)
        #         faces_normalized.append(faceArr)
        #     return faces_normalized
