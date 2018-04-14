from tensorflow.python.platform import gfile
import tensorflow as tf
import os
from sklearn.externals import joblib
from trained_model import save_model as sm
import pickle
import json

class Model:
    def __init__(self, basedir, configuration):
        self.models_available = None
        self.model = None
        self.model_path = "trained_model"
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = False
        self.sess = tf.InteractiveSession(config=self.config)
        self.base_directory = basedir
        self.configuration = configuration
        self.save_model = sm.SaveModel()
        self.configuration_details_json = None
        self.load_model_configuration()
        self.depth = configuration.depth

    def load_model_configuration(self):
        with open(os.path.join(self.base_directory, "trained_model/models.config")) as config_file:
            configuration_data = config_file.readlines()
            configuration_details = ""
            for line in configuration_data:
                configuration_details += line
            self.configuration_details_json = json.loads(str(configuration_details))


    def load_model_facenet(self):
        ''

    def restore_model(self):
        ''

    def save_model(self):
        # self.save_model.
        ''

    def create_new_session(self):
        tf.reset_default_graph()
        self.sess.close()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = False
        self.sess = tf.InteractiveSession(config=self.config)

    def load_svm_model(self, clear=True):
        if clear:
            self.create_new_session()

        model = self.configuration_details_json['svm'][str(self.depth)]
        model_file = model['model']
        svm_model = model['svm_model']
        classify_file = model['classifier']
        tensors = model['tensors']

        print("Loading Model ...", model_file)
        frozen_graph_filename = os.path.join(self.base_directory, model_file)
        with gfile.FastGFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            byte = f.read()
            graph_def.ParseFromString(byte)
            tf.import_graph_def(graph_def, name='')

        model_dict = {}
        with self.sess.as_default():
            for tensor in tensors:
                print('Fetching tensor ', tensor)
                model_dict[tensor] = tf.get_default_graph().get_tensor_by_name(tensor + ':0')
        print("Loading Model Graph for svm Complete")

        path = os.path.join(self.base_directory, svm_model)
        clf = joblib.load(path)

        print("Loading Classifier Map...")
        classifier_filename = os.path.join(self.base_directory, classify_file)
        label_pickle = open(classifier_filename, "rb")
        label_pickle_dict = pickle.load(label_pickle)
        label_pickle.close()
        model_dict['classifier'] = label_pickle_dict['classifier']
        print(model_dict['classifier'])

        return model_dict, clf

    def load_facenet_svm_model(self, clear=True):
        if clear:
            self.create_new_session()

        model = self.configuration_details_json['svm_facenet'][str(self.depth)]
        model_file = model['model']
        svm_model = model['svm_model']
        classify_file = model['classifier']
        tensors = model['tensors']

        print("Loading Model ...", model_file)
        frozen_graph_filename = os.path.join(self.base_directory, model_file)
        with gfile.FastGFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            byte = f.read()
            graph_def.ParseFromString(byte)
            tf.import_graph_def(graph_def, name='')

        model_dict = {}
        with self.sess.as_default():
            for tensor in tensors:
                print('Fetching tensor ', tensor)
                model_dict[tensor] = tf.get_default_graph().get_tensor_by_name(tensor + ':0')
        print("Loading Model Graph for facenet Complete")

        path = os.path.join(self.base_directory, svm_model)
        clf = joblib.load(path)

        print("Loading Classifier Map...")
        classifier_filename = os.path.join(self.base_directory, classify_file)
        label_pickle = open(classifier_filename, "rb")
        label_pickle_dict = pickle.load(label_pickle)
        label_pickle.close()
        model_dict['classifier'] = label_pickle_dict['classifier']
        print(model_dict['classifier'])

        return model_dict, clf

    def load_facenet_model(self, clear=True):
        if clear:
            self.create_new_session()
        model = self.configuration_details_json['facenet'][str(self.depth)]
        model_file = model['model']
        classify_file = model['classifier']
        tensors = model['tensors']

        print("Loading Model ...", model_file)
        frozen_graph_filename = os.path.join(self.base_directory, model_file)
        with gfile.FastGFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            byte = f.read()
            graph_def.ParseFromString(byte)
        tf.import_graph_def(graph_def, name='')

        model_dict = {}
        with self.sess.as_default():
            for tensor in tensors:
                print('Fetching tensor ', tensor)
                model_dict[tensor] = tf.get_default_graph().get_tensor_by_name(tensor + ':0')

        print("Loading Model Graph for facenet Complete")

        return model_dict

    def load_inception_v5_model(self, clear=True):
        if clear:
            self.create_new_session()
        # model_file = "nn_inception_v5/freeze_inception_net_v5_120_new.pb"
        # classify_file = "nn_inception_v5/classifier_inceptionv5.pickle"
        # tensors = ["input_tensor", "output_tensor", "predicted_class_probabilities", "output/BiasAdd", "embedding/BiasAdd"]
        model = self.configuration_details_json['inceptionv5'][str(self.depth)]
        model_file = model['model']
        classify_file = model['classifier']
        tensors = model['tensors']
        print("Loading Model ...", model_file)
        frozen_graph_filename = os.path.join(self.base_directory,  model_file)
        with gfile.FastGFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            byte = f.read()
            graph_def.ParseFromString(byte)
        tf.import_graph_def(graph_def, name='')
        print("Loading Model Graph for nn_inception_v5 Complete")

        model_dict = {}

        with self.sess.as_default():
            self.detection_graph = tf.get_default_graph()
            for tensor in tensors:
                print('Fetching tensor ', tensor)
                model_dict[tensor] = self.detection_graph.get_tensor_by_name(tensor + ':0')

        print("Loading Classifier Map...")

        classifier_filename = os.path.join(self.base_directory,  classify_file)
        label_pickle = open(classifier_filename, "rb")
        label_pickle_dict = pickle.load(label_pickle)
        label_pickle.close()
        model_dict['classifier'] = label_pickle_dict['classifier']
        print(model_dict['classifier'])
        return model_dict

    def load_inception_v1_model(self, clear=True):
        if clear:
            self.create_new_session()
        model = self.configuration_details_json['inceptionv1'][str(self.depth)]
        model_file = model['model']
        classify_file = model['classifier']
        tensors = model['tensors']
        print("Loading Model ...", model_file)
        # model_file = "nn_inception_v1/freeze_inception_net_v1_120_new.pb"
        # classify_file = "nn_inception_v1/classifier_inceptionv1.pickle"
        # tensors = ["input_tensor", "output_tensor", "predicted_class_probabilities", "output/BiasAdd"]
        # frozen_graph_filename = os.path.join(self.base_directory, os.path.join(self.model_path, model_file))

        frozen_graph_filename = os.path.join(self.base_directory,  model_file)
        with gfile.FastGFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            byte = f.read()
            graph_def.ParseFromString(byte)
        tf.import_graph_def(graph_def, name='')
        print("Loading Model Complete" ,model_file)

        model_dict = {}

        with self.sess.as_default():
            self.detection_graph = tf.get_default_graph()
            for tensor in tensors:
                print('Fetching tensor ', tensor)
                model_dict[tensor] = self.detection_graph.get_tensor_by_name(tensor + ':0')

        print("Loading Classifier Map...")
        classifier_filename = os.path.join(self.base_directory,  classify_file)
        label_pickle = open(classifier_filename, "rb")
        label_pickle_dict = pickle.load(label_pickle)
        label_pickle.close()
        model_dict['classifier'] = label_pickle_dict['classifier']
        print(model_dict['classifier'])
        return model_dict

    def load_nn_cnn_model(self, clear=True):
        if clear:
            self.create_new_session()

        print(self.depth)
        model_config = self.configuration_details_json['nn'][str(self.depth)]
        model_file = model_config['model']
        classify_file = model_config['classifier']
        tensors = model_config['tensors']
        print("Loading Model ...", model_file)
        # model_file = "nn_cnn/freeze_conv_net_120_new.pb"
        # classify_file = "nn_cnn/classifier.pickle"
        # tensors = ["input_tensor","output_tensor","predicted_class_probabilities","output/BiasAdd"]
        # frozen_graph_filename = os.path.join(self.base_directory, os.path.join(self.model_path, model_file))

        frozen_graph_filename = os.path.join(self.base_directory, model_file)
        with gfile.FastGFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            byte = f.read()
            graph_def.ParseFromString(byte)
        tf.import_graph_def(graph_def, name='')
        print("Loading Model Complete.", model_file)

        model_dict = {}

        with self.sess.as_default():
            self.detection_graph = tf.get_default_graph()
            for tensor in tensors:
                print('Fetching tensor ', tensor)
                model_dict[tensor] = self.detection_graph.get_tensor_by_name(tensor+':0')

        print("Loading Classifier Map...")
        classifier_filename = os.path.join(self.base_directory, classify_file)
        label_pickle = open(classifier_filename, "rb")
        label_pickle_dict = pickle.load(label_pickle)
        label_pickle.close()
        model_dict['classifier'] = label_pickle_dict['classifier']
        print(model_dict['classifier'])

        return model_dict




