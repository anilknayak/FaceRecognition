import os
import json

class Configuration:
    def __init__(self, basedir, debug=False):
        self.debug = debug
        self.basedir = basedir
        self.training_data_type = None
        # GUI
        self.detector_type = "mtcnn"
        self.recognizer_type = "nn"
        self.number_of_prediction = 1
        self.number_of_recognized = 1
        self.camera = 0
        self.video_file = None
        self.start_camera = False
        self.action = 'Recognize'
        self.frame_rate = 10
        self.video_codec = 'XVID'
        self.capture_face_dtl = 'Near to Camera'
        self.show_feature_points = False
        self.vertical_align_face = False
        self.add_padding = 0
        self.boundingbox_size_incr_by = 0
        self.down_sampling_factor = 1
        self.accumulator_status = False
        self.accumulator_frame_count = 2
        self.accumulator_weighted_status = False

        # Image
        self.height = 120
        self.width = 120

        # Data
        self.raw_data_dir = None
        self.raw_data_images = None
        self.raw_data_labels = None

        self.processed_data_dir = None
        self.processed_data_images = None
        self.processed_data_labels = None

        self.raw_data_pickle = None
        self.processed_data_pickle = None
        self.pre_processing_required = None
        self.prepare_data_from_pickle_file = None
        self.prepare_pickle_file = None
        # Pickle data Details configuration
        self.pickle_data_images = None
        self.pickle_data_labels = None

        self.report = 'No'

        # Data Augmentation
        self.data_separation = None
        self.augmentation_required = None
        self.augmented_data_size = None

        # Training Details
        self.training_method = None
        self.network_config_file = None
        self.classes = None
        self.training_size_percentage = None
        self.random_shuffle = None

        # Freeze Model Details
        self.model_name = None

        # Model Restoration Details
        self.model_restoration_required = None
        self.restore_model_graph_file = None
        self.restore_model_file = None
        self.depth = None

    def load_configuration(self, configuration_file):
        # configuration/application/app.config
        with open(os.path.join(self.basedir, configuration_file)) as config_file:
            configuration_data = config_file.readlines()
            configuration_details = ""
            for line in configuration_data:
                configuration_details += line
            configuration_details_json = json.loads(str(configuration_details))

            self.training_data_type = configuration_details_json['training_data_type']
            self.prepare_data_from_pickle_file = configuration_details_json['prepare_data_from_pickle_file']
            self.pickle_data_images = configuration_details_json['pickle_data']['images']
            self.pickle_data_labels = configuration_details_json['pickle_data']['labels']

            self.pre_processing_required = configuration_details_json['pre_processing_required']
            self.processed_data_images = configuration_details_json['processed_data_folder']['images']
            self.processed_data_labels = configuration_details_json['processed_data_folder']['labels']
            self.raw_data_images = configuration_details_json['raw_data_folder']['images']
            self.raw_data_labels = configuration_details_json['raw_data_folder']['labels']

            self.augmentation_required = configuration_details_json['augmentation_required']
            self.augmented_data_size = configuration_details_json['augmented_data_size']
            self.prepare_pickle_file = configuration_details_json['prepare_pickle_file']

            self.training_method = configuration_details_json['training_method']
            self.network_config_file = configuration_details_json['network_config_file']
            self.classes = configuration_details_json['classes']
            self.training_size_percentage = configuration_details_json['training_size_percentage']
            self.random_shuffle = configuration_details_json['random_shuffle']

            # Model Restoration Details Configuration
            self.model_restoration_required = configuration_details_json['model_file']['restoration_model_required']
            self.restore_model_graph_file = configuration_details_json['model_file']['restore_model']['model_graph']
            self.restore_model_file = configuration_details_json['model_file']['restore_model']['model']

            # After Train model save details
            # self.model_name = configuration_details_json['model_file']['model_name']

            # Data Separation
            self.data_separation = configuration_details_json['data_separation']

            # Images Dimension
            self.height = configuration_details_json['image_size']['height']
            self.width = configuration_details_json['image_size']['width']

            self.depth = configuration_details_json['image_depth']