import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# Set camera resolution. The max resolution is webcam dependent
# so change it to a resolution that is both supported by your camera
# and compatible with your monitor
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

# If you have problems running this code on MacOS X you probably have to reinstall opencv with
# qt backend because cocoa support seems to be broken:
#   brew reinstall opencv --HEAD --qith-qt
cv2.namedWindow('frame', cv2.cv.CV_WINDOW_NORMAL)
cv2.namedWindow(CALIBRATE_FORM_NAME, CV_WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
ret, frame = cap.read()
cv2.imshow('frame', frame)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
