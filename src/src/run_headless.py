import argparse
import ast
import os
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import common


def main():
    parser = argparse.ArgumentParser(description='Headless tf-pose-estimation run (saves image instead of showing window)')
    parser.add_argument('--image', type=str, default='./images/apink2.jpg')
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368',
                        help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--scales', type=str, default='[None]', help='e.g. [1.0, (1.1, 0.05)]')
    parser.add_argument('--output', type=str, default='output_pose.jpg', help='output image path')
    args = parser.parse_args()

    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.model)
    estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    image = common.read_imgfile(args.image, None, None)
    t = time.time()
    humans = estimator.inference(image, scales=scales)
    elapsed = time.time() - t

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    image_bgr = TfPoseEstimator.draw_humans(image_bgr, humans, imgcopy=False)

    cv2.imwrite(args.output, image_bgr)
    print('Saved result to %s (inference %.4f s)' % (os.path.abspath(args.output), elapsed))


if __name__ == '__main__':
    main()

