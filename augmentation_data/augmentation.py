import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm
import cv2

class Augmentation:
    def __init__(self, config):
        shift = 0.2
        self.datagen = keras.preprocessing.image.ImageDataGenerator(
                                                                        # featurewise_std_normalization=True,
                                                                        # featurewise_center=True,
                                                                        rotation_range=30,
                                                                        # width_shift_range=0.1,
                                                                        # height_shift_range=0.1,
                                                                        width_shift_range=shift,
                                                                        height_shift_range=shift,
                                                                        shear_range=0.1,
                                                                        zoom_range=0.1,
                                                                        fill_mode='nearest',
                                                                        horizontal_flip=True,
                                                                        vertical_flip=False
                                                                    )
        self.configuration = config

    def augment(self, data, flag, depth=1, reshape=True):
        if flag:
            return self.augment_data(data, depth, reshape)
        else:
            return self.rearrange_data(data, depth, reshape)

    def rearrange_data(self, data, depth, reshape):
        classes = data.classes
        training_images_dict = data.data_as_whole

        images = np.empty((0, self.configuration.height, self.configuration.width, depth))
        labels = np.empty(0, dtype='uint8')
        label_number_mapping = {}
        data_stat_augment = {}
        count = -1
        for label in tqdm(classes, desc="Rearranging Data"):
            count = count + 1
            label_number_mapping[count] = label
            images_per_class = training_images_dict[label]

            images_loaded_per_class = []
            labels_loaded_per_class = []
            for image in images_per_class:
                if reshape:
                    image_reshaped = image.reshape((self.configuration.height, self.configuration.width, depth))
                    images_loaded_per_class.append(image_reshaped)
                images_loaded_per_class.append(image)
                labels_loaded_per_class.append(count)

            images_loaded_per_class = np.asarray(images_loaded_per_class)
            labels_loaded_per_class = np.asarray(labels_loaded_per_class)
            data_stat_augment[label] = len(images_loaded_per_class)
            images = np.append(images, images_loaded_per_class, axis=0)
            labels = np.append(labels, labels_loaded_per_class, axis=0)

        data.images = images
        data.labels = labels
        data.labels_number_mapping = label_number_mapping
        data.dict_raw_data_stat = data_stat_augment
        return data

    def augment_data(self, data, depth, reshape):
        classes = data.classes
        training_images_dict = data.data_as_whole

        count = -1

        images = np.empty((0, self.configuration.height, self.configuration.width, depth))
        labels = np.empty(0, dtype='uint8')
        label_number_mapping = {}
        generate_total_images = self.configuration.augmented_data_size
        dict_augmented_data_stat = {}
        image_count = 0
        data_stat_augment = {}
        for label in tqdm(classes, desc="Augmenting Data"):
            count = count + 1
            images_to_be_augmented = []
            # labels_to_be_augmented = []
            labels_number_to_be_augmented = []
            label_number_mapping[count] = label

            images_to_be_augmented_lst = training_images_dict[label]
            images_per_label_count = len(images_to_be_augmented_lst)
            image_count = image_count + images_per_label_count
            # print("Before Augmentation number of images for label :", label , str(images_per_label_count))
            for image_per_label in images_to_be_augmented_lst:
                # print(np.shape(image_per_label))
                image_sized = cv2.resize(image_per_label, (self.configuration.height, self.configuration.width), interpolation=cv2.INTER_CUBIC)
                image_reshaped = image_sized.reshape((self.configuration.height, self.configuration.width, depth))
                images_to_be_augmented.append(image_reshaped)
                # labels_to_be_augmented.append(label)
                labels_number_to_be_augmented.append(count)

            # Images converted to numpy array for augmentation
            images_to_be_augmented = np.asarray(images_to_be_augmented)
            # labels_to_be_augmented = np.asarray(labels_to_be_augmented)
            labels_number_to_be_augmented = np.asarray(labels_number_to_be_augmented)

            # copying the images and labels to be augmented
            images_to_be_augmented_copy = np.copy(images_to_be_augmented)
            # labels_to_be_augmented_copy = np.copy(labels_to_be_augmented)
            labels_number_to_be_augmented_copy = np.copy(labels_number_to_be_augmented)

            '''
            TODO : Change the seed
            '''
            for img, cls in self.datagen.flow(images_to_be_augmented, labels_number_to_be_augmented, batch_size=len(labels_number_to_be_augmented), seed= 2 + count * 37):
                images_to_be_augmented_copy = np.append(images_to_be_augmented_copy, img, axis=0)
                labels_number_to_be_augmented_copy = np.append(labels_number_to_be_augmented_copy, cls, axis=0)

                if len(images_to_be_augmented_copy) >= generate_total_images:
                    break

            dict_augmented_data_stat[label] = len(images_to_be_augmented_copy)
            # print("After Augmentation number of images for label :", label, str(len(images_to_be_augmented_copy)))
            data_stat_augment[label] = len(images_to_be_augmented_copy)
            images = np.append(images, images_to_be_augmented_copy, axis=0)
            labels = np.append(labels, labels_number_to_be_augmented_copy, axis=0)

        # TODO Dump the class number vs label srt mapping

        print("Total Images before Augmentation : ", str(image_count))
        print("Total Images after Augmentation : ", str(len(images)))

        data.images = images
        data.labels = labels
        data.labels_number_mapping = label_number_mapping
        data.dict_augmented_data_stat = dict_augmented_data_stat
        data.augmented = True
        data.dict_augment_data_stat = dict_augmented_data_stat
        # data.images = shuffle_images
        # data.labels = shuffle_labels

        print('Data Augmentation is complete')
        return data