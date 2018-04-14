from facedetection.mtcnn import face

class FindFace:
    def __init__(self, base_directory):
        self.detector = face.Detection()

    def getfaces(self, image):
        boxes, pts = self.detector.find_faces(image)
        return boxes, pts
