'''
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
'''

import cv2, colorsys, random, time
import numpy as np

def show_webcam(mirror=False):
  cam = cv2.VideoCapture(0)
  fgbg = cv2.BackgroundSubtractorMOG2()
  fgmask = None
  ret_val, img = cam.read()
  img = img[::2, ::2, :]
  colorFrame = np.zeros(img.shape)
  cnt = 0
  t0 = time.time()
  cv2.namedWindow('test', cv2.cv.CV_WINDOW_NORMAL)
  while True:
    cnt += 1
    ret_val, img = cam.read()
    img = img[::2, ::2, :]
    if mirror: 
      img = cv2.flip(img, 1)
    fgmask = fgbg.apply(img, fgmask, 0.01)
    whiteGuy = np.where(fgmask == 255, 255, 0)
    fgmask -= np.roll(fgmask, -2, 0)

    # randHue = random.random()
#     randVal = random.random()
#     red, green, blue = colorsys.hsv_to_rgb(randHue, 1.0, randVal)
#     red = int(red*255)
#     green = int(green*255)
#     blue = int(blue*255)
#     colorFrame[0:2, :, :] = 0
#     colorFrame = np.roll(colorFrame, -2, 0)
    print cnt / (time.time() - t0)
#     colorFrame = np.where((whiteGuy == 255)[:, :, np.newaxis], 0, colorFrame)
#     rgb = np.array([red, green, blue])
#     colorFrame = np.where(fgmask[:,:,np.newaxis] == 255, rgb[np.newaxis, np.newaxis, :], colorFrame)
#     colorFrame = np.clip(colorFrame, 0, 255)
#     disp = np.copy(colorFrame)
#     disp[:, :, 1] += whiteGuy
    #fgmask = 255 - fgmask
    cv2.imshow("test", fgmask)
    if cv2.waitKey(1) == 27: 
      break  # esc to quit
  cv2.destroyAllWindows()

def main():
  show_webcam(mirror=True)

if __name__ == '__main__':
  main()
