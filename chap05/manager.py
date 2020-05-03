import logging
from multiprocessing import Value, Process, Manager
import time
from PCA9685 import *

# import pantilthat as pth
import signal
import sys
import numpy as np

from camera import PiCameraStream
from ssd_mobilenet_v3_coco import SSDMobileNet_V3_Small_Coco_PostProcessed
from pid import PIDController

logging.basicConfig()
LOGLEVEL = logging.getLogger().getEffectiveLevel()

RESOLUTION = (320, 320)

SERVO_MIN = 0
SERVO_MAX = 180

CENTER = (
    RESOLUTION[0] // 2,
    RESOLUTION[1] // 2
)

# function to handle keyboard interrupt

def signal_handler(sig, frame):
    # print a status message
    print("[INFO] You pressed `ctrl + c`! Exiting...")
    # exit
    sys.exit()


def run_detect(center_x, center_y, labels, model_cls):

    model = model_cls()

    capture_manager = PiCameraStream(resolution=RESOLUTION)
    capture_manager.start()
    capture_manager.start_overlay()

    label_idxs = model.label_to_category_index(labels)
    start_time = time.time()
    fps_counter = 0
    while not capture_manager.stopped:
        if capture_manager.frame is not None:
            frame = capture_manager.read()
            prediction = model.predict(frame)

            if not len(prediction.get('detection_boxes')):
                continue

            if any(item in label_idxs for item in prediction.get('detection_classes')):

                tracked = (
                    (i, x) for i, x in
                    enumerate(prediction.get('detection_classes'))
                    if x in label_idxs
                )
                tracked_idxs, tracked_classes = zip(*tracked)

                track_target = prediction.get('detection_boxes')[tracked_idxs[0]]    # [ymin, xmin, ymax, xmax]
                y = int(RESOLUTION[1] - ((np.take(track_target, [0, 2])).mean() * RESOLUTION[1]))
                center_y.value = y
                x = int(RESOLUTION[0] - ((np.take(track_target, [1, 3])).mean() * RESOLUTION[0]))
                center_x.value = x

                print('Box: ({}, {})'.format(x, y))

                display_name = model.category_index[tracked_classes[0]]['name']
                logging.info(f'Tracking {display_name} center_x {x} center_y {y}')

            overlay = model.create_overlay(frame, prediction)
            capture_manager.overlay_buff = overlay
            if LOGLEVEL is logging.DEBUG and (time.time() - start_time) > 1:
                fps_counter += 1
                fps = fps_counter / (time.time() - start_time)
                logging.debug(f'FPS: {fps}')
                fps_counter = 0
                start_time = time.time()


def in_range(val, start, end):
    # determine the input vale is in the supplied range
    return (val >= start and val <= end)


def set_servos(pan, tilt):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    setAngle(0, 90)
    setAngle(1, 0)
    pan_angle = 90
    tilt_angle = 45

    while True:
        pan_delta = pan.value
        tilt_delta = 0.01 * tilt.value
        pan_angle -= pan_delta
        tilt_angle += tilt_delta
        pan_angle = min(180, max(0, pan_angle))
        tilt_angle = min(120, max(30, tilt_angle))
        print('{}, {}, {}'.format(pan.value, pan_delta, pan_angle))

        # if the pan angle is within the range, pan
        if in_range(pan_angle, SERVO_MIN, SERVO_MAX):
            # pth.pan(pan_angle)
            setAngle(0, pan_angle)
        else:
            logging.info(f'pan_angle not in range {pan_angle}')

        if in_range(tilt_angle, SERVO_MIN, SERVO_MAX):
            setAngle(1, tilt_angle)
            # pth.tilt(tilt_angle)
        else:
            logging.info(f'tilt_angle not in range {tilt_angle}')

        # logging.info('Pan Angle: {}'.format(pan_angle))
        # # logging.info('Tilt Angle: {}'.format(tilt_angle))
        # print('Pan Angle: {}'.format(pan_angle))
        # # print('Tilt Angle: {}'.format(tilt_angle))
        time.sleep(0.5)

def pid_process(output, p, i, d, box_coord, origin_coord, action):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    # create a PID and initialize it
    pid = PIDController(p.value, i.value, d.value)
    pid.reset()

    # loop indefinitely
    while True:
        error = origin_coord - box_coord.value
        output.value = pid.update(error)
        logging.info(f'{action} error {error} angle: {output.value}')
        # print(f'{action} error {error} angle: {output.value}')
# ('person',)
#('orange', 'apple', 'sports ball')


def pantilt_process_manager(
    model_cls,
    labels=('person',)
):

    with Manager() as manager:
        # set initial bounding box (x, y)-coordinates to center of frame
        center_x = manager.Value('i', 0)
        center_y = manager.Value('i', 0)

        center_x.value = RESOLUTION[0] // 2
        center_y.value = RESOLUTION[1] // 2

        # pan and tilt angles updated by independent PID processes
        pan = manager.Value('i', 0)
        tilt = manager.Value('i', 0)

        # PID gains for panning

        pan_p = manager.Value('f', 0.05)
        # 0 time integral gain until inferencing is faster than ~50ms
        pan_i = manager.Value('f', 0)
        pan_d = manager.Value('f', 0)

        # PID gains for tilting
        tilt_p = manager.Value('f', 0.15)
        # 0 time integral gain until inferencing is faster than ~50ms
        tilt_i = manager.Value('f', 0.2)
        tilt_d = manager.Value('f', 0)

        detect_processr = Process(target=run_detect,
                                  args=(center_x, center_y, labels, model_cls))

        pan_process = Process(target=pid_process,
                              args=(pan, pan_p, pan_i, pan_d, center_x, CENTER[0], 'pan'))

        tilt_process = Process(target=pid_process,
                               args=(tilt, tilt_p, tilt_i, tilt_d, center_y, CENTER[1], 'tilt'))

        servo_process = Process(target=set_servos, args=(pan, tilt))

        detect_processr.start()
        pan_process.start()
        tilt_process.start()
        servo_process.start()

        detect_processr.join()
        pan_process.join()
        tilt_process.join()
        servo_process.join()


if __name__ == '__main__':
    pantilt_process_manager(SSDMobileNet_V3_Small_Coco_PostProcessed)
