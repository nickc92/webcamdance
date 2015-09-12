'''
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
'''

import cv2, colorsys, random, time, math
import numpy as np

import pyaudio, struct, time
import wave, numpy as np
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"
FREQ_CUTOFF = 400.0

audio = pyaudio.PyAudio()

rmsEMA = 0.0
lastTime = 0.0
emaTau = 2.0
lastX = 0.0
lastY = 0.0
def callback(data, frameCount, time_info, status):
    global micLevel, rmsEMA, lastTime, emaTau, lastX, lastY

    count = len(data) / 2
    format = '%dh'%(count)
    shorts = struct.unpack(format, data)
    alpha = 1.0 / (1.0 + 2 * math.pi * FREQ_CUTOFF / RATE)
    ys = []
    for i in range(len(shorts)):
        y = alpha * lastY + alpha * (shorts[i] - lastX)
        ys.append(y)
        lastX = shorts[i]        
        lastY = y
    npdat = np.array(ys)
    rms = np.sqrt(np.dot(npdat, npdat)/float(npdat.shape[0]))
    t = time.time()
    alpha = math.exp(-(t - lastTime) / emaTau)
    lastTime = t
    rmsEMA = alpha * rmsEMA + (1.0 - alpha) * rms
    micLevel = rms / rmsEMA / 2.0
    if micLevel > 1.0: micLevel = 1.0

    return (None, pyaudio.paContinue)
    
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                output=False, stream_callback=callback)


colorLastT = 0.0
colorEMATau = 3.0

def show_webcam(mirror=False):
  global colorLastT, colorEMATau
  
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
    fgmask = fgbg.apply(img, fgmask, 0.003)
    # whiteGuy = np.where(fgmask > 0, 255, 0).astype('uint8')
#     fgmask -= np.roll(fgmask, -2, 0)
    randHue = random.random()
#     #randVal = random.random()
    randVal = micLevel

#     colorFrame[0:2, :, :] = 0
#     colorFrame = np.roll(colorFrame, -2, 0)
#     print cnt / (time.time() - t0)
#     colorFrame = np.where((whiteGuy == 255)[:, :, np.newaxis], 0, colorFrame)
#     rgb = np.array([red, green, blue])
#     colorFrame = np.where(fgmask[:,:,np.newaxis] == 255, rgb[np.newaxis, np.newaxis, :], colorFrame)
#     colorFrame = np.clip(colorFrame, 0, 255)
#     disp = np.copy(colorFrame)
#     val = int(randVal * 255)
#     disp[:, :, 1] = np.where(whiteGuy > 0, val, disp[:, :, 1])
#     #fgmask = 255 - fgmask
#     print 'val:', randVal
    disp = (randVal * fgmask).astype('uint8')    
    if randVal > 0.8:
        red, green, blue = colorsys.hsv_to_rgb(randHue, 1.0, randVal)
        red = int(red*255)
        green = int(green*255)
        blue = int(blue*255)
        print 'rand hue:', randHue
        t = time.time()
        decay = math.exp(-(t - colorLastT) / colorEMATau)
        print decay
        colorLastT = t
        colorFrame = (colorFrame * decay)
        colorFrame[:, :, 0] += (1.0 - decay) * red * disp
        colorFrame[:, :, 1] += (1.0 - decay) * green * disp
        colorFrame[:, :, 2] += (1.0 - decay) * blue * disp
        colorFrame = np.clip(colorFrame, 0, 255).astype('uint8')
    
    disp2 = colorFrame.copy()
    if randVal > 0.8:
        disp2 = 150 * np.ones(disp2.shape)
    disp2[:, :, 0] += disp
    disp2[:, :, 1] += disp
    disp2[:, :, 2] += disp
    disp2 = np.clip(disp2, 0, 255)
    cv2.imshow("test", disp2)
    if cv2.waitKey(1) == 27: 
      break  # esc to quit
  cv2.destroyAllWindows()

def main():
  show_webcam(mirror=True)

if __name__ == '__main__':
  main()
