import io
import time
import picamera
import numpy as np
import cv2
import face_recognition
from PCA9685 import *
import threading

IMG_WID = 320
IMG_HEI = 240

pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)
pwm.setServoPulse(0,1500)
pwm.setServoPulse(1,1000)

stop_thread = False
pan = 1500
tilt = 1000
c_pan = 0.5
c_tilt = 0.5

def servo_control():
    while not stop_thread:
        pwm.setServoPulse(1, tilt)
        pwm.setServoPulse(0, pan)
        time.sleep(0.02)

servo_thread = threading.Thread(target=servo_control)
servo_thread.start()

try:
    camera = picamera.PiCamera()
    camera.resolution = (IMG_WID, IMG_HEI)
    # Start a preview and let the camera warm up for 2 seconds
    camera.start_preview()
    time.sleep(2)

    stream = io.BytesIO()

    for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
        stream.seek(0)
        img_bytes = stream.read()
        file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        face_locations = face_recognition.face_locations(image)
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            print(left, right, top, bottom)

            error_x = IMG_WID//2 - (left + right) // 2
            error_y = (top + bottom) // 2 - IMG_HEI//2

            delta_pan = int(error_x * c_pan)
            delta_tilt = int(error_y * c_tilt)

            print(delta_pan, delta_tilt)
            tilt = min(1500, max(500, tilt + delta_tilt))
            pan = min(2500, max(500, pan + delta_pan))
            print(pan, tilt)

        stream.seek(0)
        stream.truncate()

except KeyboardInterrupt:
    stop_thread = True
    servo_thread.join()
    pwm.setServoPulse(0,1500)
    pwm.setServoPulse(1,1000)
    print('Terminated...')
