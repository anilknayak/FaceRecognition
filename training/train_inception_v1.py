import tensorflow as tf
from tqdm import tqdm
import os
from tensorflow.python.tools import freeze_graph
import numpy as np

class TrainNeuralNetwork:
    def __init__(self, commuter, depth=1):
        self.commuter = commuter
        self.depth = depth
        self.network_configuration_file_name = ''
        self.data_obj = None
        self.model_obj = None
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = False
        self.sess = tf.InteractiveSession(config=self.config)

        self.resultwriter = self.commuter.context.resultwritter.create_result_obj('inceptionv1')

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

        # Prepare Neural Network
        self.model_obj = self.commuter.prepare_deep_network("configuration/nn_architecture/inception_net_v1_new.config", depth=self.depth)

        # Train/Validate Model
        self.train_model(self.data_obj, self.model_obj)

        # Test Model

        # Save Model

    def train_model(self, data_obj, model_obj):
        import math
        print('Starting Training ...')
        images = data_obj.training_data['images']
        labels = data_obj.training_data['labels_n']
        label_number_mapping = data_obj.labels_number_mapping
        dict_loss_stat = []
        dict_evaluation_stat = []
        saver = tf.train.Saver()

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            for i in range(self.model_obj.training_epochs):
                training_batch = self.commuter.prepare_batch(images, labels, model_obj.batch_size)
                epoch_loss = 0
                for j in tqdm(range(0, len(images), model_obj.batch_size), desc="Training Epoch Inception V1 : "+str(i+1)):
                    batch_images, batch_labels = training_batch.next_batch()
                    _, loss = self.sess.run([self.model_obj.model, self.model_obj.loss_operation], feed_dict={self.model_obj.input_tensor: batch_images,
                                                                                                              self.model_obj.output_tensor: batch_labels})
                    epoch_loss += loss

                print("Epoch Loss for epoch inception V1: ", str(i), " is ", epoch_loss)
                accuracy = self.validate_model(data_obj.validation_data['images'], data_obj.validation_data['labels_n'], model_obj, "Validation ")
                dict_loss_stat.append(epoch_loss)
                dict_evaluation_stat.append(accuracy)

                if math.floor(epoch_loss) == 0:
                    break

            self.resultwriter.export_report('loss', dict_loss_stat)
            self.resultwriter.export_report('evaluation', dict_evaluation_stat)
            data_stat = {}
            data_stat['initial'] = self.data_obj.dict_raw_data_stat
            data_stat['augment'] = self.data_obj.dict_augment_data_stat
            self.resultwriter.export_report('data_stat', data_stat)

            accuracytest = self.validate_model(data_obj.testing_data['images'], data_obj.testing_data['labels_n'], model_obj, "Testing ")

            self.save_model(saver, label_number_mapping)

    def save_model(self, saver, label_number_mapping):
        import pickle
        print('Freezing Currently Trained Model in ')
        model_dir_path = os.path.join(self.commuter.context.base_directory, self.model_obj.model_detail['model_dir'])
        # model_dir_path = "./model/"
        print(model_dir_path)
        checkpoint_prefix = os.path.join(model_dir_path, 'model.ckpt')
        print(checkpoint_prefix)
        saver.save(self.sess, checkpoint_prefix)  # ,global_step=50
        tf.train.write_graph(self.sess.graph.as_graph_def(), model_dir_path, str(self.depth)+self.model_obj.model_detail['model_graph'],
                             as_text=True)

        input_graph_path = os.path.join(model_dir_path, str(self.depth)+self.model_obj.model_detail['model_graph'])
        print("Model Graph ", input_graph_path)
        input_saver_def_path = ""
        input_binary = False
        input_checkpoint_path = checkpoint_prefix
        output_graph_path = os.path.join(model_dir_path, str(self.depth)+self.model_obj.model_detail['model_name'])
        print("Model File ", output_graph_path)
        clear_devices = False
        print("Output Nodes", self.model_obj.node_names)
        output_node_names = self.model_obj.node_names
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        initializer_nodes = ""
        freeze_graph.freeze_graph(input_graph_path,
                                  input_saver_def_path,
                                  input_binary,
                                  input_checkpoint_path,
                                  output_node_names,
                                  restore_op_name,
                                  filename_tensor_name,
                                  output_graph_path,
                                  clear_devices,
                                  initializer_nodes)

        print('Saving Classifier Mapping')
        data_temp = {}
        data_temp['classifier'] = label_number_mapping
        path = os.path.join(model_dir_path, "classifier_inceptionv1_"+str(self.depth)+".pickle")
        pickle_in = open(path, "wb")
        pickle.dump(data_temp, pickle_in)
        pickle_in.close()

        print('Deleting all the checkpoint and meta files')

        print('Model Saved Successfully')

    def validate_model(self, images, labels, model_obj, type):
        validation_batch = self.commuter.prepare_batch(images,
                                                       labels,
                                                       model_obj.batch_size)
        validation_total_image = len(images)
        acc_total = 0
        accuracy = 0
        with self.sess.as_default():
            for j in range(0, validation_total_image, self.model_obj.batch_size):
                batch_images, batch_labels = validation_batch.next_batch()
                (self.output) = self.sess.run([self.model_obj.output],
                                                           feed_dict={self.model_obj.input_tensor: batch_images,
                                                                      self.model_obj.output_tensor: -1})
                pred = np.array(np.argmax(self.output[0], axis=1).tolist())
                actual = np.array(batch_labels)

                acc_total = acc_total + np.sum(pred==actual)

            accuracy = acc_total / validation_total_image
            print(type," accuracy is [correct, total, accuracy]", acc_total, validation_total_image, accuracy)

        return accuracy