import datetime
import logging
import time
from multiprocessing import Manager, Process

import cv2
import keyboard
import numpy

from multiprocessed import DetectorAPI, get_key

logging.basicConfig(
    format="[%(asctime)s]%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG
)

logger = logging.getLogger(__file__)

threshold = 0.9


def parse_images(data):
    logger.debug("parse_images started")
    model_path = "frozen_inference_graph.pb"
    odapi = DetectorAPI(path_to_ckpt=model_path)
    time.sleep(5)
    last_box = None

    while True:
        logger.debug("parse_images got image: %s", data["time"].isoformat())
        boxes, scores, classes, num = odapi.processFrame(data["image"])
        human_boxes = [
            boxes[i]
            for i in range(len(boxes))
            if classes[i] == 1 and scores[i] > threshold
        ]
        data["boxes"] = human_boxes
        logger.debug("boxes: %s", human_boxes)

        if human_boxes:
            box = human_boxes[0]
            if last_box:
                key = get_key(last_box, box)
                logging.debug("key: %s", key)
                if key:
                    data["text"] = key.value
                    keyboard.press_and_release(key.value)
                else:
                    data["text"] = None
            last_box = box
        else:
            last_box = None
            data["text"] = None


def put_text(image, text, position):
    font = cv2.FONT_HERSHEY_DUPLEX
    shadow_offset = 2
    shadow_position = (position[0] + shadow_offset, position[1] + shadow_offset)
    cv2.putText(image, text, shadow_position, font, 1, (0, 0, 0), 2)
    cv2.putText(image, text, position, font, 1, (255, 255, 255), 2)


def show_images(data):
    logger.debug("show_images started")
    cv2.imshow("preview", numpy.ones((1, 1, 1), numpy.uint8) * 255)
    while True:
        time.sleep(1 / 60.0)  # limit fps
        if "image" in data:
            img = data["image"].copy()
            for box in data["boxes"]:
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            if "text" in data:
                put_text(img, data["text"], (30, 30))
            cv2.imshow("preview", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def get_images(data):
    logger.debug("get_images started")
    cap = cv2.VideoCapture(0)
    while True:
        img = cap.read()[1].copy()
        img = cv2.flip(img, 1)
        data["image"] = img
        data["time"] = datetime.datetime.now()

    cap.release()


def main():
    with Manager() as manager:
        data = manager.dict(boxes=[])

        processes = [
            Process(target=get_images, args=(data,), daemon=True),
            Process(target=parse_images, args=(data,), daemon=True),
            Process(target=show_images, args=(data,), daemon=True),
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()


if __name__ == "__main__":
    main()
