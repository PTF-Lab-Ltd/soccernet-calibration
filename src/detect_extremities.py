import argparse
import copy
import json
import os.path
import random
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

from .soccerpitch import SoccerPitch


def generate_class_synthesis(semantic_mask, radius):
    """
    This function selects for each class present in the semantic mask, a set of circles that cover most of the semantic
    class blobs.
    :param semantic_mask: a image containing the segmentation predictions
    :param radius: circle radius
    :return: a dictionary which associates with each class detected a list of points ( the circles centers)
    """
    buckets = dict()
    kernel = np.ones((5, 5), np.uint8)
    semantic_mask = cv.erode(semantic_mask, kernel, iterations=1)
    for k, class_name in enumerate(SoccerPitch.lines_classes):
        mask = semantic_mask == k + 1
        if mask.sum() > 0:
            disk_list = synthesize_mask(mask, radius)
            if len(disk_list):
                buckets[class_name] = disk_list

    return buckets


def join_points(point_list, maxdist):
    """
    Given a list of points that were extracted from the blobs belonging to a same semantic class, this function creates
    polylines by linking close points together if their distance is below the maxdist threshold.
    :param point_list: List of points of the same line class
    :param maxdist: minimal distance between two polylines.
    :return: a list of polylines
    """
    polylines = []

    if not len(point_list):
        return polylines
    head = point_list[0]
    tail = point_list[0]
    polyline = deque()
    polyline.append(point_list[0])
    remaining_points = copy.deepcopy(point_list[1:])

    while len(remaining_points) > 0:
        min_dist_tail = 1000
        min_dist_head = 1000
        best_head = -1
        best_tail = -1
        for j, point in enumerate(remaining_points):
            dist_tail = np.sqrt(np.sum(np.square(point - tail)))
            dist_head = np.sqrt(np.sum(np.square(point - head)))
            if dist_tail < min_dist_tail:
                min_dist_tail = dist_tail
                best_tail = j
            if dist_head < min_dist_head:
                min_dist_head = dist_head
                best_head = j

        if min_dist_head <= min_dist_tail and min_dist_head < maxdist:
            polyline.appendleft(remaining_points[best_head])
            head = polyline[0]
            remaining_points.pop(best_head)
        elif min_dist_tail < min_dist_head and min_dist_tail < maxdist:
            polyline.append(remaining_points[best_tail])
            tail = polyline[-1]
            remaining_points.pop(best_tail)
        else:
            polylines.append(list(polyline.copy()))
            head = remaining_points[0]
            tail = remaining_points[0]
            polyline = deque()
            polyline.append(head)
            remaining_points.pop(0)
    polylines.append(list(polyline))
    return polylines


def get_line_extremities(buckets, maxdist, width, height):
    """
    Given the dictionary {lines_class: points}, finds plausible extremities of each line, i.e the extremities
    of the longest polyline that can be built on the class blobs,  and normalize its coordinates
    by the image size.
    :param buckets: The dictionary associating line classes to the set of circle centers that covers best the class
    prediction blobs in the segmentation mask
    :param maxdist: the maximal distance between two circle centers belonging to the same blob (heuristic)
    :param width: image width
    :param height: image height
    :return: a dictionary associating to each class its extremities
    """
    extremities = dict()
    for class_name, disks_list in buckets.items():
        polyline_list = join_points(disks_list, maxdist)
        max_len = 0
        longest_polyline = []
        for polyline in polyline_list:
            if len(polyline) > max_len:
                max_len = len(polyline)
                longest_polyline = polyline
        extremities[class_name] = [
            {'x': longest_polyline[0][1] / width,
                'y': longest_polyline[0][0] / height},
            {'x': longest_polyline[-1][1] / width,
                'y': longest_polyline[-1][0] / height}
        ]
    return extremities


def get_support_center(mask, start, disk_radius, min_support=0.1):
    """
    Returns the barycenter of the True pixels under the area of the mask delimited by the circle of center start and
    radius of disk_radius pixels.
    :param mask: Boolean mask
    :param start: A point located on a true pixel of the mask
    :param disk_radius: the radius of the circles
    :param min_support: proportion of the area under the circle area that should be True in order to get enough support
    :return: A boolean indicating if there is enough support in the circle area, the barycenter of the True pixels under
     the circle
    """
    x = int(start[0])
    y = int(start[1])
    support_pixels = 1
    result = [x, y]
    xstart = x - disk_radius
    if xstart < 0:
        xstart = 0
    xend = x + disk_radius
    if xend > mask.shape[0]:
        xend = mask.shape[0] - 1

    ystart = y - disk_radius
    if ystart < 0:
        ystart = 0
    yend = y + disk_radius
    if yend > mask.shape[1]:
        yend = mask.shape[1] - 1

    for i in range(xstart, xend + 1):
        for j in range(ystart, yend + 1):
            dist = np.sqrt(np.square(x - i) + np.square(y - j))
            if dist < disk_radius and mask[i, j] > 0:
                support_pixels += 1
                result[0] += i
                result[1] += j
    support = True
    if support_pixels < min_support * np.square(disk_radius) * np.pi:
        support = False

    result = np.array(result)
    result = np.true_divide(result, support_pixels)

    return support, result


def synthesize_mask(semantic_mask, disk_radius):
    """
    Fits circles on the True pixels of the mask and returns those which have enough support : meaning that the
    proportion of the area of the circle covering True pixels is higher that a certain threshold in order to avoid
    fitting circles on alone pixels.
    :param semantic_mask: boolean mask
    :param disk_radius: radius of the circles
    :return: a list of disk centers, that have enough support
    """
    mask = semantic_mask.copy().astype(np.uint8)
    points = np.transpose(np.nonzero(mask))
    disks = []
    while len(points):

        start = random.choice(points)
        dist = 10.
        success = True
        while dist > 1.:
            enough_support, center = get_support_center(
                mask, start, disk_radius)
            if not enough_support:
                bad_point = np.round(center).astype(np.int32)
                cv.circle(
                    mask, (bad_point[1], bad_point[0]), disk_radius, (0), -1)
                success = False
            dist = np.sqrt(np.sum(np.square(center - start)))
            start = center
        if success:
            disks.append(np.round(start).astype(np.int32))
            cv.circle(mask, (disks[-1][1], disks[-1][0]), disk_radius, 0, -1)
        points = np.transpose(np.nonzero(mask))

    return disks


class SegmentationNetwork:
    def __init__(self, model_file, mean_file, std_file, num_classes=29, width=640, height=360):
        file_path = Path(model_file).resolve()
        model = nn.DataParallel(deeplabv3_resnet50(
            pretrained=False, num_classes=num_classes))
        self.init_weight(model, nn.init.kaiming_normal_,
                         nn.BatchNorm2d, 1e-3, 0.1,
                         mode='fan_in')
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(str(file_path), map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model = model.to(self.device)
        file_path = Path(mean_file).resolve()
        self.mean = np.load(str(file_path))
        file_path = Path(std_file).resolve()
        self.std = np.load(str(file_path))
        self.width = width
        self.height = height

    def init_weight(self, feature, conv_init, norm_layer, bn_eps, bn_momentum,
                    **kwargs):
        for name, m in feature.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                conv_init(m.weight, **kwargs)
            elif isinstance(m, norm_layer):
                m.eps = bn_eps
                m.momentum = bn_momentum
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def analyse_image(self, image):
        """
        Process image and perform inference, returns mask of detected classes
        :param image: BGR image
        :return: predicted classes mask
        """
        img = cv.resize(image, (self.width, self.height),
                        interpolation=cv.INTER_LINEAR)
        img = np.asarray(img, np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device).unsqueeze(0)

        cuda_result = self.model.forward(img.float())
        output = cuda_result['out'].data[0].cpu().numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('-i', '--images', type=str, required=True,
                        help='Path to the images to detect field on')
    parser.add_argument('-p', '--prediction', required=True, type=str,
                        help="Path to the prediction folder")
    parser.add_argument('--masks', required=False, type=bool,
                        default=False, help='Save masks in prediction directory')
    parser.add_argument('--resolution_width', required=False, type=int, default=640,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=360,
                        help='height resolution of the images')
    parser.add_argument('--prefix', required=False, type=str, default='jpg',
                        help='Prefix for the input image files')
    args = parser.parse_args()

    lines_palette = [0, 0, 0]
    for line_class in SoccerPitch.lines_classes:
        lines_palette.extend(SoccerPitch.palette[line_class])

    resources_dir = Path(__file__).resolve(
    ).parent.parent.parent / "resources"
    calib_net = SegmentationNetwork(
        resources_dir / "soccer_pitch_segmentation.pth",
        resources_dir / "mean.npy",
        resources_dir / "std.npy")

    images_dir = args.images
    if not os.path.exists(images_dir):
        print("Invalid images directory path !")
        exit(-1)

    output_prediction_folder = args.prediction
    if not os.path.exists(output_prediction_folder):
        print("Invalid prediction directory path !")
        exit(-1)

    frame_path_list = sorted(
        list(Path(images_dir).glob(str("*.") + args.prefix)))

    with tqdm(enumerate(frame_path_list), total=len(frame_path_list), ncols=160) as t:
        for i, frame_path in t:
            prediction = dict()
            count = 0

            image = cv.imread(str(frame_path))
            semlines = calib_net.analyse_image(image)
            if args.masks:
                mask = Image.fromarray(semlines.astype(np.uint8)).convert('P')
                mask.putpalette(lines_palette)
                mask_file = os.path.join(
                    output_prediction_folder, frame_path.with_suffix(".png").name)
                mask.save(mask_file)
            skeletons = generate_class_synthesis(semlines, 6)
            extremities = get_line_extremities(
                skeletons, 40, args.resolution_width, args.resolution_height)

            prediction = extremities
            count += 1

            prediction_file = os.path.join(
                output_prediction_folder, frame_path.with_suffix(".json").name)
            with open(prediction_file, "w") as f:
                json.dump(prediction, f, indent=4)