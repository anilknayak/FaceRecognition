import os
import json
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

class NNModel:
    def __init__(self, basedir, configuration):
        self.number_of_class_label = None
        self.base_dir = basedir
        self.network_json = None
        self.nn_architecture = None
        self.configuration = configuration
        # Training Details
        self.training_epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.stopping_loss_threshold_from_previous = None
        self.optimizer_function = None
        self.loss_function = None
        self.regularization_beta = None
        self.nn_input_size = None
        self.nn_output_size = None
        self.tensor_name = None
        self.input_height = None
        self.input_width = None
        self.resize_required = None
        # self.dropout_flag = None
        # self.dropout_percentage = None

        # tensors
        self.output = None
        self.input_tensor = None
        self.output_tensor = None
        self.output_tensor_one_hot = None
        self.node_names = "output_tensor,input_tensor,loss,output_tensor_one_hot,output/BiasAdd"
        self.correct_prediction = None
        self.accuracy_operation = None
        self.regularizers = None
        self.cross_entropy = None
        self.optimizer = None
        self.loss_operation = None
        # self.nodes = []

        # Model
        self.model_detail = None
        self.model = None

    def load_nn_architecture(self, nn_arch_path):
        nn_arch_folder = os.path.join(self.base_dir, nn_arch_path)
        with open(nn_arch_folder) as net:
            net_data = net.readlines()
            net_details_str = ""
            for line in net_data:
                net_details_str += line
            self.network_json = json.loads(str(net_details_str))

        self.training_epochs = self.network_json['training']['training_steps']
        self.batch_size = self.network_json['training']['batch_size']
        self.learning_rate = self.network_json['training']['learning_rate']
        self.stopping_loss_threshold_from_previous = self.network_json['training']['stopping_loss_threshold_from_previous']
        self.optimizer_function = self.network_json['training']['optimizer']
        self.loss_function = self.network_json['training']['loss']

        # self.dropout_flag = self.network_json['training']['dropout']
        # self.dropout_percentage = self.network_json['training']['dropout_percentage']

        self.regularization_beta = self.network_json['training']['regularization_beta']
        self.nn_input_size = self.network_json['training']['nn_input_size']
        self.nn_output_size = self.network_json['training']['nn_output_size']
        self.tensor_name = self.network_json['training']['tensor_name']

        self.input_height = self.network_json['image']['height']
        self.input_width = self.network_json['image']['width']
        self.resize_required = self.network_json['image']['resize_required']
        self.number_of_class_label = self.network_json['training']['class_label_no']
        self.model_detail = self.network_json['model_file']

        self.nn_architecture = self.network_json['deep_neural_network']

    def build_nn_model(self, depth=1):
        classes_number = self.number_of_class_label
        neural_network_dict = self.nn_architecture
        layers_op = []
        self.input_tensor = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, depth], name='input_tensor')
        self.output_tensor = tf.placeholder(tf.int32, (None), name='output_tensor')
        self.output_tensor_one_hot = tf.one_hot(self.output_tensor, classes_number, name='output_tensor_one_hot')
        # self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        layers_op.append(self.input_tensor)
        position_of_output_layer_without_softmax = None



        for layer in neural_network_dict:
            # if not layer['type'] == 'flat':
            #     self.node_names = self.node_names + "," + layer['name']

            print('preparing layer ', layer)
            """
                If you are adding any type of the deep learning methods just mention as below
            """

            if layer['type'] == 'conv':
                layers_op.append(self.conv_layer(layers_op[-1], layer))
            elif layer['type'] == 'maxpool':
                layers_op.append(self.maxpool_layer(layers_op[-1], layer))
            elif layer['type'] == 'dense':
                layers_op.append(self.dense_layer(layers_op[-1], layer))
                self.node_names = self.node_names + "," + layer['name'] + "/BiasAdd"
            elif layer['type'] == 'flat':
                layers_op.append(self.flat(layers_op[-1], layer))
            elif layer['type'] == 'output':
                layers_op.append(self.output_detection_layer(layers_op[-1], layer))
            elif layer['type'] == 'dropout':
                layers_op.append(self.dropout(layers_op[-1], layer))
            # elif layer['type'] == 'softmax':
            #     if position_of_output_layer_without_softmax is not None:
            #         position_of_output_layer_without_softmax += 1
            #     layers_op.append(self.output_detection_layer(layers_op[-1], layer))

        # print("Node Name" , self.node_names)
        self.output = layers_op[-1] # logits

        predicted_class_probabilities = tf.nn.softmax(self.output, name="predicted_class_probabilities")
        predicted_class_without_softmax = tf.argmax(input=self.output, axis=1, name="predicted_class_without_softmax")
        predicted_class_with_softmax = tf.argmax(input=predicted_class_probabilities, axis=1, name="predicted_class_with_softmax")
        self.node_names = self.node_names + ",predicted_class_probabilities,predicted_class_without_softmax,predicted_class_with_softmax"

        self.prepare_loss_function()
        self.prepare_optimizer_function()

        self.correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.output_tensor_one_hot, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # self.nodes = [self.input_tensor.name, self.output_tensor.name, self.loss_operation.name]
        # self.process_node_names()

    def conv_layer(self, prev_layer, layer):
        # convolution = None
        # TODO Activation

        if 'regularization' in layer.keys() and layer['regularization']:
            regularizer = self.regularization(layer)

            convolution = tf.layers.conv2d(
                inputs=prev_layer,
                filters=layer['filters'],
                kernel_size=layer['kernel'],
                padding=layer['padding'],
                strides=layer['strides'],
                activation=tf.nn.relu,
                name=layer['name'],
                kernel_regularizer=regularizer)
        else:
            convolution = tf.layers.conv2d(
                inputs=prev_layer,
                filters=layer['filters'],
                kernel_size=layer['kernel'],
                padding=layer['padding'],
                strides=layer['strides'],
                activation=tf.nn.relu,
                name=layer['name'])

        return convolution

    def regularization(self, layer):
        if 'regularization_func' in layer.keys() and layer['regularization_func'] == "l2":
            return tf.contrib.layers.l2_regularizer(scale=layer['regularization_scale'])

    def flat(self,prev_layer, layer):
        return tf.contrib.layers.flatten(prev_layer)

    def maxpool_layer(self, prev_layer, layer):
        max_pooling = tf.layers.max_pooling2d(inputs=prev_layer, pool_size=layer['pool_size'], strides=layer['strides'], name=layer['name'])
        return max_pooling


    def dense_layer(self, prev_layer, layer):
        dense = tf.layers.dense(inputs=prev_layer,
                                units=layer['units'],
                                activation=tf.nn.relu,
                                name=layer['name'])
        # dropout = tf.layers.dropout(
        #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        return dense


    def output_detection_layer(self, prev_layer, layer):
        logits = tf.layers.dense(inputs=prev_layer, units=layer['units'], name=layer['name'])

        return logits

    def inception(self, prev_layer, layer):
        inception_model_name = layer['name']
        inception_blocks = layer['block']
        parallel_blocks_inception = []
        for inception_block in inception_blocks:
            block_layer = [prev_layer]
            for layer_inception in inception_block:
                layer_inception['name'] = inception_model_name + "_" + layer_inception['name']
                self.node_names = self.node_names + "," + layer_inception['name']
                if layer_inception['type'] == 'conv':
                    block_layer.append(self.conv_layer(block_layer[-1], layer_inception))
                elif layer_inception['type'] == 'maxpool':
                    block_layer.append(self.maxpool_layer(block_layer[-1], layer_inception))
            parallel_blocks_inception.append(block_layer[-1])
            # np.append(parallel_blocks_inception,block_layer[-1])

        # Filter Concentration
        # concatenate all the feature maps and hit them with a relu
        # inception_layer = tf.nn.relu(tf.concat(3, parallel_blocks_inception))
        inception_layer = tf.nn.relu(tf.concat(parallel_blocks_inception, 3), name=inception_model_name)
        return inception_layer

    def dropout(self, prev_layer, layer):
        # TODO Apply mode
        dropout = tf.layers.dropout(inputs=prev_layer, rate=layer['dropout_percentage'], name=layer['name'])
        return dropout

    # def softmax(self, prev_layer, layer):
    #     predictions = {
    #         # Generate predictions (for PREDICT and EVAL mode)
    #         "classes": tf.argmax(input=prev_layer, axis=1),
    #         # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    #         # `logging_hook`.
    #         "probabilities": tf.nn.softmax(prev_layer, name="softmax_tensor")
    #         "probabilities_softmax": tf.argmax(input=tf.nn.softmax(prev_layer, name="softmax_tensor"), axis=1)
    #     }
    #
    #     return predictions

    def prepare_optimizer_function(self):
        if self.optimizer_function == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="optimizer")
        if self.optimizer_function == "gradient_descent":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name="optimizer")

        self.model = self.optimizer.minimize(self.loss_operation)

    def prepare_loss_function(self):
        if self.loss_function == "softmax_cross_entropy":
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                                         labels=self.output_tensor_one_hot,
                                                                         name="softmax")
            self.loss_operation = tf.reduce_mean(self.cross_entropy, name="loss")

        elif self.loss_function == "sparse_softmax_cross_entropy":
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output,
                                                                                labels=self.output_tensor_one_hot,
                                                                                name="softmax")
            self.loss_operation = tf.reduce_mean(self.cross_entropy, name="loss")

        elif self.loss_function == "softmax_cross_entropy_v2":
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,
                                                                            labels=self.output_tensor_one_hot,
                                                                            name="softmax")
            self.loss_operation = tf.reduce_mean(self.cross_entropy, name="loss")

        if self.regularizers is not None:
            self.loss_operation = tf.reduce_mean(self.loss_operation + self.regularization_beta * self.regularizers)
