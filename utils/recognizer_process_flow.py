import datetime
import os
import commuter as comm
import numpy as np

class RecognizerProcessFlow():
    def __init__(self, val, debug_flag=False):
        # Thread.__init__(self)
        self.start_camera = False
        self.stop = True
        self.UI_started = False
        self.debug = debug_flag
        self.val = val
        self.commuter = comm.Commuter('GUI', debug_flag)
        self.capture_dir = None
        self.folder_created = False
        self.video_codec_is_ready = False
        self.capture_dir_final = None
        self.writter = None
        self.report_writter = False
        self.resultwritter1 = None

        self.total_frame = 0
        self.total_frame_face_detected = 0
        self.total_frame_face_recognized = 0
        self.frame_wise_recognition_list = []
        self.method = ''
        self.depth = 1
        self.video_number = 1
        self.start = False

    def stop_process(self):
        self.start = False
        self.commuter.context.configuration.action = 'Stop'
        self.commuter.context.configuration.start_camera = False
        self.commuter.stop_camera()
        self.folder_created = False
        if self.video_codec_is_ready:
            if self.writter is not None:
                self.writter.release()
            self.video_codec_is_ready = False


        if self.commuter.context.configuration.report == 'Yes':
            print('dumping video analysis data')
            dict2 = {}
            dict2['total_frame'] = self.total_frame
            dict2['total_frame_face_detected'] = self.total_frame_face_detected
            dict2['total_frame_face_recognized'] = self.total_frame_face_recognized
            dict2['frame_wise_recognition_list'] = self.frame_wise_recognition_list

            if self.resultwritter1:
                self.resultwritter1.export_report('video_analysis', dict2, video_number=self.video_number)
                self.video_number = self.video_number + 1

        self.total_frame = 0
        self.total_frame_face_detected = 0
        self.total_frame_face_recognized = 0
        self.frame_wise_recognition_list = []

    def create_writter(self):
        type = None
        if self.commuter.context.configuration.recognizer_type == 'nn':
            type = 'cnn'
        elif self.commuter.context.configuration.recognizer_type == 'inception_v1':
            type = 'inception_v1'
        elif self.commuter.context.configuration.recognizer_type == 'inception_v5':
            type = 'inception_v5'
        elif self.commuter.context.configuration.recognizer_type == 'svm':
            type = 'svm'
        elif self.commuter.context.configuration.recognizer_type == 'svm_facenet':
            type = 'svm_facenet'
        elif self.commuter.context.configuration.recognizer_type == 'facenet':
            type = 'facenet'

        print("writter", type)
        self.resultwritter1 = self.commuter.context.resultwritter.create_result_obj(type)
        self.report_writter = True


    def process(self):

        frame = self.commuter.get_input('camera')
        print(np.shape(frame))
        return frame

        # while self.start:
        #     frame = None
        #     faces = None
        #
        #     if self.commuter.context.configuration.start_camera:
        #         frame = self.commuter.get_input('camera')
        #
        #     if self.commuter.context.configuration.action == 'Display':
        #         if frame is not None:
        #             display.output(frame, None)
        #         else:
        #             display.stop()
        #
        #     elif self.commuter.context.configuration.action == 'Recognize':
        #
        #         if self.commuter.context.configuration.report == 'Yes' and not self.report_writter:
        #             self.create_writter()
        #
        #         # self.app.processEvents()
        #         if frame is not None:
        #             self.total_frame = self.total_frame + 1
        #             # self.app.processEvents()
        #             faces = self.commuter.detect(frame)
        #             if self.debug and faces is not None:
        #                 print('Detected Faces ', len(faces))
        #
        #             if self.commuter.context.configuration.report == 'Yes' and faces is not None and len(faces) > 0:
        #                 self.total_frame_face_detected = self.total_frame_face_detected + 1
        #
        #             # self.app.processEvents()
        #             faces = self.commuter.recognize(faces)
        #             if self.debug:
        #                 print('Recognized Faces ', len(faces))
        #
        #             if self.commuter.context.configuration.report == 'Yes' and faces is not None and len(faces) > 0:
        #                 self.total_frame_face_recognized = self.total_frame_face_recognized + 1
        #                 for face in faces:
        #                     labels = face.labels_pred
        #                     probs = face.prob_pred
        #
        #                     dict1 = {}
        #                     dict1['label'] = labels[0]
        #                     dict1['prob'] = probs[0]
        #                     dict1['frame_number'] = self.total_frame
        #                     self.frame_wise_recognition_list.append(dict1)
        #
        #             # self.app.processEvents()
        #             faces = self.commuter.post_processing(faces, self.commuter.context.configuration)
        #
        #             if self.debug:
        #                 print('Post Processing Faces ', len(faces))
        #
        #             # self.app.processEvents()
        #             processed_frame = self.commuter.display(faces, frame,
        #                                                     'All Faces',
        #                                                     self.commuter.context.configuration.show_feature_points)
        #             display.output(processed_frame, faces)
        #
        #     elif self.commuter.context.configuration.action == 'Capture':
        #
        #         if self.capture_dir is None:
        #             self.capture_dir = os.path.join(os.path.dirname(__file__),"data/captured/")
        #
        #         if not self.folder_created:
        #             folder_name = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '-')
        #             folder_name = folder_name.split(".")[0]
        #             self.capture_dir_final = self.capture_dir + folder_name
        #             if not os.path.isdir(self.capture_dir_final):
        #                 os.mkdir(self.capture_dir_final, 0o777)
        #
        #             self.folder_created = True
        #
        #         if not self.video_codec_is_ready:
        #             self.writter = self.commuter.get_video_capture_obj(self.capture_dir_final, self.commuter.context.configuration.video_codec)
        #             self.video_codec_is_ready = True
        #
        #         if frame is not None:
        #             if self.writter is not None and frame is not None:
        #                 self.writter.write(frame)
        #
        #             frame1 = np.copy(frame)
        #
        #             # self.app.processEvents()
        #             faces = self.commuter.detect(frame)
        #
        #             # self.app.processEvents()
        #             faces = self.commuter.faces_area_calculation(faces)
        #
        #             # self.app.processEvents()
        #             processed_frame = self.commuter.display(faces, frame,
        #                                                     self.commuter.context.configuration.capture_face_dtl,
        #                                                     self.commuter.context.configuration.show_feature_points)
        #             display.output(processed_frame, None)
        #
        #             if faces is not None and len(faces) > 0:
        #                 # self.app.processEvents()
        #                 self.commuter.save_faces(frame1, faces, self.capture_dir_final, self.commuter.context.configuration.capture_face_dtl)
        #
        #     else:
        #         ''