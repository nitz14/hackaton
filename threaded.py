import threading
import time
from enum import Enum
from queue import Empty, LifoQueue, Queue

import cv2
import keyboard
import numpy as np
import tensorflow as tf


class Key(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name("image_tensor:0")
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            "detection_boxes:0"
        )
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            "detection_scores:0"
        )
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            "detection_classes:0"
        )
        self.num_detections = self.detection_graph.get_tensor_by_name(
            "num_detections:0"
        )

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        # start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [
                self.detection_boxes,
                self.detection_scores,
                self.detection_classes,
                self.num_detections,
            ],
            feed_dict={self.image_tensor: image_np_expanded},
        )
        # end_time = time.time()

        # print("Elapsed Time:", end_time - start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (
                int(boxes[0, i, 0] * im_height),
                int(boxes[0, i, 1] * im_width),
                int(boxes[0, i, 2] * im_height),
                int(boxes[0, i, 3] * im_width),
            )

        return (
            boxes_list,
            scores[0].tolist(),
            [int(x) for x in classes[0].tolist()],
            int(num[0]),
        )

    def close(self):
        self.sess.close()
        self.default_graph.close()


def threaded_function(kolejka, do_rozpoznania):
    model_path = "frozen_inference_graph.pb"
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.9
    while True:
        img = kolejka.get()
        boxes, scores, classes, num = odapi.processFrame(img)
        ludzie = [
            boxes[i]
            for i in range(len(boxes))
            if classes[i] == 1 and scores[i] > threshold
        ]
        if len(ludzie) > 1:
            print("BLAD! TYLKO JEDNA OSOBA MA GRAC!")
        elif len(ludzie) == 1:
            do_rozpoznania.put_nowait(ludzie[0])

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break


def rozpoznaj(do_rozpoznania, do_sterowania):
    last_box = do_rozpoznania.get()
    while True:
        box = do_rozpoznania.get()
        if last_box and box:
            key = get_key(last_box, box)
        if key:
            do_sterowania.put_nowait(key)
        last_box = box


def get_key(p_box, c_box):
    tolerancja_bledu = 0.2

    p_height = abs(p_box[0] - p_box[2])
    p_with = abs(p_box[1] - p_box[3])
    delta = p_with * tolerancja_bledu
    delta_h = p_height * tolerancja_bledu
    print("parsuje")

    if c_box[1] < p_box[1] - delta and c_box[3] > p_box[3] + delta:
        return Key.UP
    elif c_box[1] < p_box[1] - delta:
        return Key.LEFT
    elif c_box[3] > p_box[3] + delta:
        return Key.RIGHT
    elif c_box[0] > p_box[0] + delta_h:
        return Key.DOWN

    return None


def steruj(do_sterowania):
    while True:
        key = do_sterowania.get()
        if key == Key.DOWN:
            keyboard.press_and_release(key.value)
            keyboard.press_and_release(key.value)
            keyboard.press_and_release(key.value)
            keyboard.press_and_release(key.value)
            keyboard.press_and_release(key.value)
            keyboard.press_and_release(key.value)
            keyboard.press_and_release(key.value)
            keyboard.press_and_release(key.value)
            keyboard.press_and_release(key.value)

        keyboard.press_and_release(key.value)
        print(key)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    kolejka = LifoQueue(maxsize=1)
    do_rozpoznania = Queue()
    do_sterowania = Queue()
    thread = threading.Thread(
        target=threaded_function, args=(kolejka, do_rozpoznania), daemon=True
    )
    thread.start()

    thread_rozpoznania = threading.Thread(
        target=rozpoznaj, args=(do_rozpoznania, do_sterowania), daemon=True
    )
    thread_rozpoznania.start()

    thread_sterowania = threading.Thread(
        target=steruj, args=(do_sterowania,), daemon=True
    )
    thread_sterowania.start()

    while True:
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = cv2.flip(img, 1)

        try:
            kolejka.get_nowait()
            kolejka.get_nowait()
        except Empty:
            pass
        kolejka.put_nowait(img.copy())
