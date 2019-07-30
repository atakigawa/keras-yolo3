import sys
import os
import argparse
from yolo import YOLO
import cv2 as cv
from PIL import Image
import numpy as np


def detect_images(yolo, images_dir):
    wnd_name = 'image'
    cv.namedWindow(wnd_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(wnd_name, 700, 700)

    for f in os.listdir(images_dir):
        if not f.lower().endswith('jpg') or \
                f.lower().endswith('jpeg'):
            continue
        try:
            image = Image.open(os.path.join(images_dir, f))
        except Exception:
            print(f'Failed to open {f}! Try again!')
            sys.exit(1)

        print(f'detect: {f}')
        result_pil_image = yolo.detect_image(image)
        result_np_image = cv.cvtColor(
                np.asarray(result_pil_image), cv.COLOR_RGB2BGR)
        cv.imshow(wnd_name, result_np_image)
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
    yolo.close_session()


FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path"))
    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path"))
    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path"))
    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num")))
    parser.add_argument(
        '--images_dir', required=True, help='Images dir')
    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='[Optional] confidence threshold')
    parser.add_argument(
        '--nms_threshold', type=float, default=0.5,
        help='[Optional] nms threshold')

    FLAGS = parser.parse_args()

    print("Image detection mode")
    detect_images(YOLO(**vars(FLAGS)), FLAGS.images_dir)
