import os
import sys
import argparse

class TrainFaceNet:
    def __init__(self, commuter, depth=1):
        self.depth = depth

    def train(self):
        print("Training with facenet")
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
        export_dir = os.path.join(base_dir, "facenet/src")
        command1 = "export PYTHONPATH="+export_dir
        print(command1)
        # os.system(command1)
        command1 = command1 +"; echo $PYTHONPATH ; pwd"
        command2 = "cd facenet/"
        # os.system(command1)
        print(command2)
        command3 = "rm -rf "+base_dir+"/data/images/processed120_facenet"
        print(command3)
        command4 = "mkdir " + base_dir + "/data/images/processed120_facenet"
        print(command4)
        command5 = "python src/align/align_dataset_mtcnn.py "+base_dir+"/data/images/processed120 "+base_dir+"/data/images/processed120_facenet --image_size 160 --margin 32 --random_order"
        # os.system(command2)
        print(command5)
        command6 = "python src/classifier.py TRAIN "+base_dir+"/data/images/processed120_facenet "+base_dir+"/facenet/model/20170512-110547.pb "+base_dir+"/trained_model/facenet/lfw_classifier.pkl"
        # os.system(command3)
        print(command6)
        command = command1 + " ; " + command2 + " ; " + command3 + " ; " + command4 + " ; " + command5 + " ; " + command6
        # os.system(command)

