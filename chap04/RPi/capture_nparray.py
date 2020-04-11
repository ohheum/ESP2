import time
import picamera
import numpy as np
import cv2

with picamera.PiCamera() as camera:
    camera.resolution = (320, 240)
    camera.framerate = 24
    time.sleep(2)
    output = np.empty((240, 320, 3), dtype=np.uint8)
    camera.capture(output, 'rgb')

    cv2.imwrite('image.jpg', output)
    img_left = output[:, :160, :]
    img_right_tr = np.transpose(output[:, 160:, :], (1, 0, 2))

    cv2.imwrite('left.jpg', img_left)
    cv2.imwrite('right_tr.jpg', img_right_tr)
