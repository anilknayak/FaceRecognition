import cv2 as cv


camera = cv.VideoCapture('2018-03-05_05-34-51.372072.avi')

while(True):

    ret,frame = camera.read()

    if ret:
        print('reading')
        cv.imshow("Video",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv.destroyAllWindows()