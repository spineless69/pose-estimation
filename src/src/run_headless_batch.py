import argparse
import ast
import os
import time
from glob import glob

import cv2

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import common


def main():
    parser = argparse.ArgumentParser(description='Headless batch pose estimation (saves results to a folder)')
    parser.add_argument('--input-dir', type=str, default='./images', help='folder with input images')
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368',
                        help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--scales', type=str, default='[None]', help='e.g. [1.0, (1.1, 0.05)]')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='folder to save results')
    args = parser.parse_args()

    scales = ast.literal_eval(args.scales)

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob(os.path.join(args.input_dir, ext)))
    image_paths.sort()

    if not image_paths:
        print('No images found in %s' % os.path.abspath(args.input_dir))
        return

    w, h = model_wh(args.model)
    estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, img_name)

        image = common.read_imgfile(img_path, None, None)
        t = time.time()
        humans = estimator.inference(image, scales=scales)
        elapsed = time.time() - t

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image_bgr = TfPoseEstimator.draw_humans(image_bgr, humans, imgcopy=False)
        cv2.imwrite(out_path, image_bgr)
        print('Saved %s (%.4f s)' % (os.path.abspath(out_path), elapsed))


if __name__ == '__main__':
    main()

