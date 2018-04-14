

class Data:
    def __init__(self):
        self.data_as_whole = None
        # Below are the Dictionary which hold images and labels

        self.training_data = None
        self.testing_data = None
        self.validation_data = None
        self.number_of_class_label = None

        self.cross_validation_window_data = None

        self.classes = None
        self.classes_count = None
        self.class_label_wise_images = None
        self.class_label_images = None

        # after Augmentation
        self.augmented = False
        self.images = None
        self.labels = None
        self.labels_number_mapping = None
        self.dict_augment_data_stat = None

        # After Shuffled
        self.shuffled_images = None
        self.shuffled_labels = None

        # Statistics
        self.dict_raw_data_stat = None
        self.dict_augmented_data_stat = None
        self.loss_stat = None
        self.accuracy_stat = None
