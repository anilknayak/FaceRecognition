import os
import commuter as comm
import cv2
import datetime
import numpy as np
class ProcessImage:
    def __init__(self):
        self.commuter = comm.Commuter('GUI')
        self.from_dir = self.commuter.context.base_directory+"/data/images/images_to_be_processed/original"
        self.to_dir = self.commuter.context.base_directory+"/data/images/images_to_be_processed/processed"

    def read_data(self):
        files = []
        dirs = []
        files_dir = os.listdir(self.from_dir)
        for value in files_dir:
            if os.path.isdir(os.path.join(self.from_dir, value)):
                dirs.append(value)
            else:
                files.append(value)

        print("Number of Folders", dirs)
        print("Number of Files", files)
        idnor_files = ['Thumbs.db', '.DS_Store']
        counter = 0
        # Process Dir Images First
        for dir in dirs:
            dir_path = os.path.join(self.from_dir, dir)
            files_int = os.listdir(dir_path)
            print("Number of Files in dir ",dir_path, len(files_int))
            for file in files_int:
                if file in idnor_files:
                    continue
                print("write complete for file ",file)
                image_path = os.path.join(dir_path,file)
                image = cv2.imread(image_path)

                print("Before Resize", np.shape(image))
                print("After Resize", np.shape(cv2.resize(image,(200,200))))
                print("After Resize", np.shape(cv2.resize(image, (200, 200)) , ))

                print(np.shape(image))
                if np.shape(image)[0] > 0:
                    faces = self.commuter.context.detect.get_without_process_faces(image)

                    for face in faces:
                        counter = counter + 1
                        dirname = os.path.join(self.to_dir, dir)
                        if not os.path.exists(dirname):
                            os.mkdir(dirname)

                        uniq_filename = str(datetime.datetime.now().date()) + '_' + str(
                                datetime.datetime.now().time()).replace(':', '-') + str(counter)+".jpg"
                        filename = os.path.join(dirname, uniq_filename)
                        cv2.imwrite(filename, face.image)

sd = ProcessImage()
sd.read_data()