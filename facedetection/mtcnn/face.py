import tensorflow as tf
import numpy as np
from facedetection.mtcnn import detect_face as df, mtcnn as mtcnn
from scipy import misc

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None

gpu_memory_fraction = 0.3

class Detection:
    # face detection parameters

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return mtcnn.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, ptr = df.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        # for bb in bounding_boxes:
        #     face = Face()
        #     face.container_image = image
        #     face.bounding_box = np.zeros(4, dtype=np.int32)
        #
        #     img_size = np.asarray(image.shape)[0:2]
        #     face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
        #     face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
        #     face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
        #     face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
        #     cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        #     face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
        #
        #     faces.append(face)

        return bounding_boxes, ptr

