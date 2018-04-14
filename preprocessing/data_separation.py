from sklearn.utils import shuffle
import numpy as np

class DataSeparation:
    def __init__(self, config):
        self.configuration = config

    def data_shuffling(self, data):
        '''
        :param data: Data Object Should have Images List in images and Label Number List in labels
        :return: Data Object updated with shuffled_images, shuffled_labels dataset
        '''

        print('Shuffling of data starts')
        images = data.images
        labels = data.labels
        shuffle_images, shuffle_labels = shuffle(np.array(images), np.array(labels))

        data.shuffled_images = shuffle_images
        data.shuffled_labels = shuffle_labels

        print('Data Shuffling is complete')

        return data

    def separate_data(self, data):
        '''
        :param data: Data Object Should have Images List in images and Label Number List in labels
        :return: Data Object updated with training_data, validation_data, testing_data dataset
        '''

        self.data_shuffling(data)
        flag = True
        images = None
        labels = None
        if data.shuffled_images is not None:
            images = data.shuffled_images
            labels = data.shuffled_labels
            flag = False
        elif data.images is not None:
            images = data.images
            labels = data.labels
            flag = False
        else:
            assert (flag), 'No data present to divide into training, validation and testing sets'

        print('Before data Separation started, data statistics')
        print('Number of Images #', str(len(images)))
        print('Number of Labels #', str(len(labels)))
        total_images = len(labels)

        training_sample_size = int(total_images * (int(self.configuration.training_size_percentage) / 100))
        remaining = total_images - training_sample_size
        testing_size_start = training_sample_size + int(remaining / 2)
        validation_size_start = testing_size_start + int(remaining - int(remaining / 2))

        training = {}
        training_sample_images = images[0:training_sample_size]
        # training_sample_labels_number = shuffle_labels[0:training_sample_size]
        training_sample_labels = labels[0:training_sample_size]
        training['images'] = training_sample_images
        training['labels'] = []
        training['labels_n'] = training_sample_labels

        testing = {}
        testing_sample_images = images[training_sample_size:testing_size_start]
        # testing_sample_labels_number = shuffle_labels[training_sample_size:total_images]
        testing_sample_labels = labels[training_sample_size:testing_size_start]
        testing['images'] = testing_sample_images
        testing['labels'] = []
        testing['labels_n'] = testing_sample_labels

        validation = {}
        validation_sample_images = images[testing_size_start:validation_size_start]
        # testing_sample_labels_number = shuffle_labels[training_sample_size:total_images]
        validation_sample_labels = labels[testing_size_start:validation_size_start]
        validation['images'] = validation_sample_images
        validation['labels'] = []
        validation['labels_n'] = validation_sample_labels

        data.training_data = training
        data.testing_data = testing
        data.validation_data = validation

        print('Data Separation Completed.')
        return data
