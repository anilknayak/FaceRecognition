import os
import commuter as comm
import tensorflow as tf

class Train:
    def __init__(self, method, depth=1):
        print('start')
        self.commuter = comm.Commuter('TRAIN', False)
        if method == 'SVM':
            self.train_svm(self.commuter, depth)
        elif method == 'NN_CNN':
            self.train_nn(self.commuter, depth)
        elif method == 'NN_INCEPTION_V1':
            self.train_nn_inception_v1(self.commuter, depth)
        elif method == 'NN_INCEPTION_V5':
            self.train_nn_inception_v5(self.commuter, depth)
        elif method == 'FACENET':
            self.train_facenet(self.commuter, depth)
        elif method == 'SVM_FACENET':
            self.train_svm_facenet(self.commuter, depth)

    def train_facenet(self, commuter, depth):
        from training import train_facenet as fn
        facenet = fn.TrainFaceNet(commuter, depth)
        facenet.train()

    def train_svm(self, commuter, depth):
        from training import train_embedding_svm as tsmv
        svm = tsmv.TrainEmbeddingSVM(commuter, depth)
        svm.train()
        tf.reset_default_graph()
        svm.sess.close()

    def train_knn(self, commuter, depth):
        from Development.training import train_embedding_knn as tknn
        knn = tknn.TrainEmbeddingKNN(commuter, depth)
        knn.train()

    def train_nn(self, commuter, depth):
        from training import train_nn as tnn
        nn = tnn.TrainNeuralNetwork(commuter, depth)
        nn.train()
        tf.reset_default_graph()
        nn.sess.close()

    def train_nn_inception_v1(self, commuter, depth):
        from training import train_inception_v1 as nn_inception
        nn = nn_inception.TrainNeuralNetwork(commuter, depth)
        nn.train()
        tf.reset_default_graph()
        nn.sess.close()

    def train_nn_inception_v5(self, commuter, depth):
        from training import train_inception_v5 as nn_inception
        nn = nn_inception.TrainNeuralNetwork(commuter, depth)
        nn.train()
        tf.reset_default_graph()
        nn.sess.close()

    def train_svm_facenet(self, commuter, depth):
        from training import train_embedding_svm_facenet as svm_facenet
        svm_facenet_obj = svm_facenet.TrainEmbeddingSVMFaceNet(commuter, depth)
        svm_facenet_obj.train()
        tf.reset_default_graph()
        svm_facenet_obj.sess.close()

# train0 = Train('FACENET', depth = 3)
# train1 = Train('NN_CNN', depth = 1)
# train2 = Train('NN_INCEPTION_V1', depth = 1)
# train3 = Train('NN_INCEPTION_V5', depth = 1)
train4 = Train('SVM', depth = 1)



