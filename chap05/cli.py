# -*- coding: utf-8 -*-

"""Console script for rpi_deep_pantilt."""
# import logging
import sys
import time
import click
import numpy as np
import face_recognition
from camera import PiCameraStream

def run_detect(capture_manager):
    # LOGLEVEL = logging.getLogger().getEffectiveLevel()

    start_time = time.time()
    fps_counter = 0
    while not capture_manager.stopped:
        if capture_manager.frame is not None:

            frame = capture_manager.read()
            face_locations = face_recognition.face_locations(frame)
            for top, right, bottom, left in face_locations:
                print(top, right, bottom, left)

            # prediction = model.predict(frame)
            # overlay = model.create_overlay(
            #     frame, prediction)
            # capture_manager.overlay_buff = overlay
            # if LOGLEVEL <= logging.INFO:
            #     fps_counter += 1
            #     if (time.time() - start_time) > 1:
            #         fps = fps_counter / (time.time() - start_time)
            #         logging.info(f'FPS: {fps}')
            #         fps_counter = 0
            #         start_time = time.time()


def detect():
    # level = logging.getLevelName(loglevel)
    # logging.getLogger().setLevel(level)
    #
    # if edge_tpu:
    #     model = SSDMobileNet_V3_Coco_EdgeTPU_Quant()
    # else:
    #     model = SSDMobileNet_V3_Small_Coco_PostProcessed()

    capture_manager = PiCameraStream(resolution=(320, 320))
    capture_manager.start()
    capture_manager.start_overlay()
    try:
        run_detect(capture_manager)
    except KeyboardInterrupt:
        capture_manager.stop()

# def face_detect(loglevel, edge_tpu):
#     level = logging.getLevelName(loglevel)
#     logging.getLogger().setLevel(level)
#
#     if edge_tpu:
#         model =  FaceSSD_MobileNet_V2_EdgeTPU()
#         pass
#     else:
#         model = FaceSSD_MobileNet_V2()
#
#     capture_manager = PiCameraStream(resolution=(320, 320))
#     capture_manager.start()
#     capture_manager.start_overlay()
#     try:
#         run_detect(capture_manager, model)
#     except KeyboardInterrupt:
#         capture_manager.stop()


# def list_labels(loglevel):
#     level = logging.getLevelName(loglevel)
#     logging.getLogger().setLevel(level)
#     model = SSDMobileNet_V3_Small_Coco_PostProcessed()
#     print('You can detect / track the following objects:')
#     print([x['name'] for x in model.category_index.values()])
#
#
# def track(label, loglevel, edge_tpu):
#     level = logging.getLevelName(loglevel)
#     logging.getLogger().setLevel(level)
#     if edge_tpu:
#         model_cls =  SSDMobileNet_V3_Coco_EdgeTPU_Quant
#     else:
#         model_cls = SSDMobileNet_V3_Small_Coco_PostProcessed
#
#     return pantilt_process_manager(model_cls, labels=(label,))
#
# def face_track(loglevel, edge_tpu):
#     level = logging.getLevelName(loglevel)
#     logging.getLogger().setLevel(level)
#
#     if edge_tpu:
#         model_cls = FaceSSD_MobileNet_V2_EdgeTPU
#     else:
#         model_cls = FaceSSD_MobileNet_V2
#
#     return pantilt_process_manager(model_cls, labels=('face',))
#
#
# def pantilt(loglevel):
#     level = logging.getLevelName(loglevel)
#     logging.getLogger().setLevel(level)
#     return pantilt_test()

if __name__ == "__main__":
    detect()
