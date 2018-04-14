import tensorflow as tf
from tqdm import tqdm
import os
from tensorflow.python.tools import freeze_graph
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

class TrainEmbeddingSVM:
    def __init__(self, commuter, depth=1):
        self.commuter = commuter
        self.depth = depth
        self.network_configuration_file_name = ''
        self.data_obj = None
        self.model_obj = None
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = False
        self.sess = tf.InteractiveSession(config=self.config)

        self.resultwriter = self.commuter.context.resultwritter.create_result_obj('svm')

    def train(self):
        # Loading Configuration
        self.commuter.load_configuration("configuration/application/app.config")

        # Load Data Raw Data if Pre Processing Required
        # Load pre processed data if pre processing already done
        self.data_obj = self.commuter.load_data()

        # Pre Process Data
        if self.commuter.context.configuration.pre_processing_required:
            actions = ['fetchface', 'normalize', 'resize', 'reshape']
            self.data_obj = self.commuter.pre_processing(self.data_obj, actions, 'TRAIN')
        else:
            actions = ['normalize', 'resize']
            self.data_obj = self.commuter.pre_processing(self.data_obj, actions, 'TRAIN', depth=self.depth)

        # data Augmentation
        self.data_obj = self.commuter.data_augmentation(self.data_obj, flag=True, depth=self.depth, reshape=True)

        # divide data for training and testing
        self.data_obj = self.commuter.data_separation(self.data_obj, 'NORMAL')

        # If model restoration required TODO

        # Prepare Neural Network to fetch embedding details
        self.model_obj = self.commuter.load_model_embedding("configuration/nn_architecture/cnn_svm.config", False)

        # Train/Validate Model
        self.train_model(self.data_obj, self.model_obj)

        # Test Model

        # Save Model

    def train_model(self, data_obj, model_obj):
        print('Starting Training ...')
        images = data_obj.training_data['images']
        labels = data_obj.training_data['labels_n']

        embedding_lst = []
        label_lst = []

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            training_batch = self.commuter.prepare_batch(images, labels, 10)
            for j in tqdm(range(0, len(images), 10), desc="Fetching Embedding Details"):
                batch_images, batch_labels = training_batch.next_batch()

                for m in np.asarray(batch_labels):
                    label_lst.append(np.squeeze(m))

                (self.embedding) = self.sess.run([model_obj['embedding/BiasAdd']],
                                                         feed_dict={model_obj['input_tensor']: batch_images,
                                                                    model_obj['output_tensor']: -1})
                for s in self.embedding[0]:
                    embedding_lst.append(np.asarray(np.squeeze(s)))

        print("SVM Classifier Training Starts")
        self.clf = svm.SVC(kernel='rbf', probability=True, gamma=0.001, C=1)
        self.clf.fit(embedding_lst, label_lst)
        print("SVM Classifier Validation Starts")
        self.validate_model(data_obj.validation_data['images'], data_obj.validation_data['labels_n'], model_obj, "Validation ")

        print("SVM Classifier Training Complete and Saving the Trained model")
        model_dir_path = os.path.join(self.commuter.context.base_directory, "trained_model/svm/freeze_nn_svm_"+str(self.depth)+".pkl")
        joblib.dump(self.clf, model_dir_path)
        print("SVM Classifier Trained model save complete")

        print('Saving Classifier Mapping')
        pickle_path = os.path.join(self.commuter.context.base_directory, "trained_model/svm")
        import pickle
        data_temp = {}
        data_temp['classifier'] = data_obj.labels_number_mapping
        path = os.path.join(pickle_path, "classifier_nn_svm_"+str(self.depth)+".pickle")
        pickle_in = open(path, "wb")
        pickle.dump(data_temp, pickle_in)
        pickle_in.close()


    def validate_model(self, images, labels, model_obj, type):
        validation_batch = self.commuter.prepare_batch(images,
                                                       labels,
                                                       10)
        validation_total_image = len(images)
        acc_total = 0

        with self.sess.as_default():
            for j in range(0, validation_total_image, 10):
                batch_images, batch_labels = validation_batch.next_batch()
                embedding_lst = []
                label_lst = []
                for m in np.asarray(batch_labels):
                    label_lst.append(np.squeeze(m))

                (self.embedding) = self.sess.run([model_obj['embedding/BiasAdd']],
                                                             feed_dict={model_obj['input_tensor']: batch_images,
                                                                        model_obj['output_tensor']: -1})

                for s in self.embedding[0]:
                    embedding_lst.append(np.asarray(np.squeeze(s)))


                for g in range(len(embedding_lst)):
                    first_emb = embedding_lst[g]
                    predicted = self.clf.predict_proba([first_emb])
                    sorted_probability_index = sorted(range(len(predicted[0])),
                                                           key=lambda i: predicted[0][i])
                    predictions_indx = sorted_probability_index[-1:][::-1]
                    actual = label_lst[g]

                    print("actual, prediction", actual, predictions_indx[0])
                    acc_total = acc_total + np.sum(actual==predictions_indx[0])


            accuracy = acc_total / validation_total_image
            print(type," accuracy is [correct, total, accuracy]", acc_total, validation_total_image, accuracy)