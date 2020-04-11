from picamera import PiCamera
from time import sleep
camera = PiCamera()
# camera.resolution = (1024, 768)
# camera.rotation = 180
camera.start_preview()
sleep(10)
camera.capture('foo.jpg')
camera.stop_preview()
