'''
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
'''

import cv2

def show_webcam(mirror=False):
  cam = cv2.VideoCapture(0)
  fgbg = cv2.BackgroundSubtractorMOG2()
  fgmask = None
  while True:
    ret_val, img = cam.read()
    if mirror: 
      img = cv2.flip(img, 1)
    fgmask = fgbg.apply(img, fgmask, 0.03)
    cv2.imshow('my webcam', fgmask)
    print img.shape, img.dtype
    if cv2.waitKey(1) == 27: 
      break  # esc to quit
  cv2.destroyAllWindows()

def main():
  show_webcam(mirror=True)

if __name__ == '__main__':
  main()
