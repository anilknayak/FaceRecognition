import os
import commuter as comm
import cv2
import numpy as np
from tqdm import tqdm

commuter = comm.Commuter('GUI')
commuter.load_configuration("configuration/application/app.config")

classes, classes_count = commuter.context.data_loader.find_classes_for_training(commuter.context.configuration.processed_data_images)
MAX_HEIGHT = 0
MAX_WIDTH = 0
MIN_HEIGHT = 9999
MIN_WIDTH = 9999
TOTAL_IMAGE = 0
TOTAL_IMAGE_P = 0
for cls in classes:
    image_dir_path_c = os.path.join(commuter.context.configuration.processed_data_images+"120", cls)
    os.mkdir(image_dir_path_c)
    image_dir_path = os.path.join(commuter.context.configuration.processed_data_images,cls)
    images_per_labels = os.listdir(image_dir_path)
    TOTAL_IMAGE = TOTAL_IMAGE + len(images_per_labels)

    for image in images_per_labels:
        image_path = os.path.join(image_dir_path, image)
        image_file = cv2.imread(image_path)

        faces = commuter.detect(image_file)


        counter = 0
        if len(faces) > 0:
            counter = counter + 1
            image_dir_path_w = os.path.join(image_dir_path_c, str(counter)+"_"+image)

            face = faces[0].image
            # gray_scale_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # image_between_0_and_1 = gray_scale_image / 255.0
            # image_between_0_and_1 = image_between_0_and_1 - 0.5
            # normalized_image_between_ng_1_and_po_1 = image_between_0_and_1 * 2.0
            # resized_face = cv2.resize(normalized_image_between_ng_1_and_po_1, (120, 120), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(image_dir_path_w, face)

            TOTAL_IMAGE_P = TOTAL_IMAGE_P + 1
            h, w, _ = np.shape(faces[0].image)
            if MAX_HEIGHT < h:
                MAX_HEIGHT = h

            if MAX_WIDTH < w:
                MAX_WIDTH = w

            if MIN_HEIGHT < h:
                MIN_HEIGHT = h

            if MIN_WIDTH < w:
                MIN_WIDTH = w

print("TOTAL_IMAGE",TOTAL_IMAGE)
print("TOTAL_IMAGE_P",TOTAL_IMAGE_P)
print("MAX_HEIGHT",MAX_HEIGHT)
print("MAX_WIDTH",MAX_WIDTH)
print("Min_HEIGHT",MIN_HEIGHT)
print("Min_WIDTH",MIN_WIDTH)
