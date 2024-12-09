import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .soccerpitch import SoccerPitch


class Line2D:
    def __init__(self, point1, point2):
        """Initialize a line from two points.

        Args:
            point1 (tuple/list): First point (x1, y1)
            point2 (tuple/list): Second point (x2, y2)
        """
        self.p1 = np.array(point1)
        self.p2 = np.array(point2)

        # Calculate line coefficients (ax + by + c = 0)
        self.direction = self.p2 - self.p1
        self.a = -self.direction[1]
        self.b = self.direction[0]
        self.c = np.cross(self.p1, self.p2)

        # Normalize coefficients
        norm = np.sqrt(self.a**2 + self.b**2)
        self.a /= norm
        self.b /= norm
        self.c /= norm

    def point_distance(self, point):
        """Calculate perpendicular distance from a point to the line.

        Args:
            point (tuple/list): Point coordinates (x, y)

        Returns:
            float: Perpendicular distance from point to line
        """
        return abs(self.a * point[0] + self.b * point[1] + self.c)


class LineSegment2D:
    def __init__(self, start_point, end_point):
        """Initialize a line segment from start and end points.

        Args:
            start_point (tuple/list): Start point coordinates (x1, y1)
            end_point (tuple/list): End point coordinates (x2, y2)
        """
        self.start = np.array(start_point)
        self.end = np.array(end_point)
        self.length = np.linalg.norm(self.end - self.start)

    def sample_points(self, num_points=100):
        """Generate evenly spaced points along the line segment.

        Args:
            num_points (int): Number of points to generate

        Returns:
            numpy.ndarray: Array of points along the line segment
        """
        t = np.linspace(0, 1, num_points)
        points = np.outer(1-t, self.start) + np.outer(t, self.end)
        return points

    def get_endpoints(self):
        """Return the endpoints of the line segment.

        Returns:
            tuple: (start_point, end_point)
        """
        return self.start, self.end


def calculate_segment_to_line_error(segment, line, num_samples=100):
    """Calculate reprojection error between a line segment and an infinite line.

    Args:
        segment (LineSegment2D): Line segment
        line (Line2D): Infinite line
        num_samples (int): Number of sample points to use

    Returns:
        dict: Dictionary containing various error metrics
    """
    # Sample points along the segment
    points = segment.sample_points(num_samples)

    # Calculate distances from each point to the line
    distances = np.array([line.point_distance(point) for point in points])

    # Calculate error metrics
    rms_error = np.sqrt(np.mean(distances**2))
    max_error = np.max(distances)
    mean_error = np.mean(distances)

    # Calculate endpoint errors
    start_point, end_point = segment.get_endpoints()
    start_error = line.point_distance(start_point)
    end_error = line.point_distance(end_point)

    return {
        'rms_error': rms_error,
        'max_error': max_error,
        'mean_error': mean_error,
        'start_point_error': start_error,
        'end_point_error': end_error
    }


def distance(line1, line2):
    """
    Computes euclidian distance between 2D points
    :param point1
    :param point2
    :return: euclidian distance between point1 and point2
    """

    # Create two similar but not identical lines

    segment = LineSegment2D([float(line2[0]["x"]), float(line2[0]["y"])], [
        float(line2[1]["x"]), float(line2[1]["y"])])
    line = Line2D([float(line1[0]["x"]), float(line1[0]["y"])], [
        float(line1[1]["x"]), float(line1[1]["y"])])
    errors = calculate_segment_to_line_error(segment, line)
    return errors['mean_error']


def mirror_labels(lines_dict):
    """
    Replace each line class key of the dictionary with its opposite element according to a central projection by the
    soccer pitch center
    :param lines_dict: dictionary whose keys will be mirrored
    :return: Dictionary with mirrored keys and same values
    """
    mirrored_dict = dict()
    for line_class, value in lines_dict.items():
        mirrored_dict[SoccerPitch.symetric_classes[line_class]] = value
    return mirrored_dict


def evaluate_detection_prediction(detected_lines, groundtruth_lines, threshold=2.):
    """
    Evaluates the prediction of extremities. The extremities associated to a class are unordered. The extremities of the
    "Circle central" element is not well-defined for this task, thus this class is ignored.
    Computes confusion matrices for a level of precision specified by the threshold.
    A groundtruth extremity point is correctly classified if it lies at less than threshold pixels from the
    corresponding extremity point of the prediction of the same class.
    Computes also the euclidian distance between each predicted extremity and its closest groundtruth extremity, when
    both the groundtruth and the prediction contain the element class.

    :param detected_lines: dictionary of detected lines classes as keys and associated predicted extremities as values
    :param groundtruth_lines: dictionary of annotated lines classes as keys and associated annotated points as values
    :param threshold: distance in pixels that distinguishes good matches from bad ones
    :return: confusion matrix, per class confusion matrix & per class localization errors
    """
    confusion_mat = np.zeros((2, 2), dtype=np.float32)
    per_class_confusion = {}
    errors_dict = {}
    detected_classes = set(detected_lines.keys())
    groundtruth_classes = set(groundtruth_lines.keys())

    if "Circle central" in groundtruth_classes:
        groundtruth_classes.remove("Circle central")
    if "Circle central" in detected_classes:
        detected_classes.remove("Circle central")

    false_positives_classes = detected_classes - groundtruth_classes
    for false_positive_class in false_positives_classes:
        false_positives = len(detected_lines[false_positive_class])
        confusion_mat[0, 1] += false_positives
        per_class_confusion[false_positive_class] = np.array(
            [[0., false_positives], [0., 0.]])

    false_negatives_classes = groundtruth_classes - detected_classes
    for false_negatives_class in false_negatives_classes:
        false_negatives = len(groundtruth_lines[false_negatives_class])
        confusion_mat[1, 0] += false_negatives
        per_class_confusion[false_negatives_class] = np.array(
            [[0., 0.], [false_negatives, 0.]])

    common_classes = detected_classes - false_positives_classes

    for detected_class in common_classes:

        detected_points = detected_lines[detected_class]

        groundtruth_points = groundtruth_lines[detected_class]

        groundtruth_extremities = [
            groundtruth_points[0], groundtruth_points[-1]]
        predicted_extremities = [detected_points[0], detected_points[-1]]
        per_class_confusion[detected_class] = np.zeros((2, 2))

        dist = distance(groundtruth_extremities, predicted_extremities)

        errors_dict[detected_class] = dist

        if dist < threshold:
            confusion_mat[0, 0] += 1
            per_class_confusion[detected_class][0, 0] += 1
        else:
            # treat too far detections as false positives
            confusion_mat[0, 1] += 1
            per_class_confusion[detected_class][0, 1] += 1

    return confusion_mat, per_class_confusion, errors_dict


def scale_points(points_dict, s_width, s_height):
    """
    Scale points by s_width and s_height factors
    :param points_dict: dictionary of annotations/predictions with normalized point values
    :param s_width: width scaling factor
    :param s_height: height scaling factor
    :return: dictionary with scaled points
    """
    line_dict = {}
    for line_class, points in points_dict.items():
        scaled_points = []
        for point in points:
            new_point = {'x': point['x'] *
                         (s_width-1), 'y': point['y'] * (s_height-1)}
            scaled_points.append(new_point)
        if len(scaled_points):
            line_dict[line_class] = scaled_points
    return line_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('-i', '--images', type=str, required=True,
                        help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('-p', '--prediction',
                        required=True, type=str,
                        help="Path to the prediction folder")
    parser.add_argument('-t', '--threshold', default=10, required=False, type=int,
                        help="Accuracy threshold in pixels")
    parser.add_argument('--resolution_width', required=False, type=int, default=960,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=540,
                        help='height resolution of the images')
    args = parser.parse_args()

    accuracies = []
    precisions = []
    recalls = []
    errors = []
    confusions = []
    per_class_confs = []
    dict_errors = {}
    per_class_confusion_dict = {}

    images_dir = args.images
    if not os.path.exists(images_dir):
        print("Invalid images directory path !")
        exit(-1)

    annotation_files = [f for f in sorted(
        os.listdir(images_dir)) if ".json" in f]

    frame_indices = []

    with tqdm(enumerate(annotation_files), total=len(annotation_files), ncols=160) as t:
        for i, annotation_file in t:
            frame_index = annotation_file.split(".")[0]
            annotation_file = os.path.join(
                args.images, annotation_file)
            prediction_file = os.path.join(
                args.prediction, f"{frame_index}.json")

            if not os.path.exists(prediction_file):
                accuracies.append(0.)
                precisions.append(0.)
                recalls.append(0.)
                continue

            frame_indices.append(frame_index)
            with open(annotation_file, 'r') as f:
                line_annotations = json.load(f)

            with open(prediction_file, 'r') as f:
                predictions = json.load(f)

            predictions = scale_points(
                predictions, args.resolution_width, args.resolution_height)
            line_annotations = scale_points(
                line_annotations, args.resolution_width, args.resolution_height)

            img_prediction = predictions
            img_groundtruth = line_annotations
            confusion1, per_class_conf1, reproj_errors1 = evaluate_detection_prediction(img_prediction,
                                                                                        img_groundtruth,
                                                                                        args.threshold)
            confusion2, per_class_conf2, reproj_errors2 = evaluate_detection_prediction(img_prediction,
                                                                                        mirror_labels(
                                                                                            img_groundtruth),
                                                                                        args.threshold)

            accuracy1, accuracy2 = 0., 0.
            if confusion1.sum() > 0:
                accuracy1 = confusion1[0, 0] / confusion1.sum()

            if confusion2.sum() > 0:
                accuracy2 = confusion2[0, 0] / confusion2.sum()

            if accuracy1 > accuracy2:
                accuracy = accuracy1
                confusion = confusion1
                per_class_conf = per_class_conf1
                reproj_errors = reproj_errors1
            else:
                accuracy = accuracy2
                confusion = confusion2
                per_class_conf = per_class_conf2
                reproj_errors = reproj_errors2

            confusions.append(confusion)
            per_class_confs.append(per_class_conf)

            accuracies.append(accuracy)
            if confusion[0, :].sum() > 0:
                precision = confusion[0, 0] / (confusion[0, :].sum())
                precisions.append(precision)
            if (confusion[0, 0] + confusion[1, 0]) > 0:
                recall = confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])
                recalls.append(recall)

            for line_class, error in reproj_errors.items():
                if line_class in dict_errors.keys():
                    dict_errors[line_class].append(error)
                else:
                    dict_errors[line_class] = [error]
            errors.append(reproj_errors)

            for line_class, confusion_mat in per_class_conf.items():
                if line_class in per_class_confusion_dict.keys():
                    per_class_confusion_dict[line_class] += confusion_mat
                else:
                    per_class_confusion_dict[line_class] = confusion_mat

    mRecall = np.mean(recalls)
    sRecall = np.std(recalls)
    medianRecall = np.median(recalls)
    print(
        f" On SoccerNet  set, recall mean value : {mRecall * 100:2.2f}% with standard deviation of {sRecall * 100:2.2f}% and median of {medianRecall * 100:2.2f}%")

    mPrecision = np.mean(precisions)
    sPrecision = np.std(precisions)
    medianPrecision = np.median(precisions)
    print(
        f" On SoccerNet set, precision mean value : {mPrecision * 100:2.2f}% with standard deviation of {sPrecision * 100:2.2f}% and median of {medianPrecision * 100:2.2f}%")

    mAccuracy = np.mean(accuracies)
    sAccuracy = np.std(accuracies)
    medianAccuracy = np.median(accuracies)
    print(
        f" On SoccerNet set, accuracy mean value : {mAccuracy * 100:2.2f}% with standard deviation of {sAccuracy * 100:2.2f}% and median of {medianAccuracy * 100:2.2f}%")

    results = {
        "threshold": str(args.threshold),
        #  "dataset_errors": {
        "mean_recall": str(mRecall),
        "std_recall": str(sRecall),
        "median_recall": str(medianRecall),
        # "mean_precision": str(mPrecision),
        # "std_precision": str(sPrecision),
        # "median_precision": str(medianPrecision),
        # "mean_accuracy": str(mAccuracy),
        # "std_accuracy": str(sAccuracy),
        # "median_accuracy": str(medianAccuracy)
    }

    # print(dict_errors)

    for line_class, confusion_mat in per_class_confusion_dict.items():
        class_accuracy = confusion_mat[0, 0] / confusion_mat.sum()
        class_recall = confusion_mat[0, 0] / \
            (confusion_mat[0, 0] + confusion_mat[1, 0])
        class_precision = confusion_mat[0, 0] / \
            (confusion_mat[0, 0] + confusion_mat[0, 1])
        print(
            f"For class {line_class}, accuracy of {class_accuracy * 100:2.2f}%, precision of {class_precision * 100:2.2f}%  and recall of {class_recall * 100:2.2f}%")
        results[line_class] = {}
        if line_class in dict_errors.keys():
            results[line_class]["recall"] = str(class_recall)
            if line_class != "Line unknown" and line_class != "Goal unknown":
                results[line_class]["mean_reproj_error"] = np.mean(
                    dict_errors[line_class])

    results["file_errors"] = {}

    # for k, v in dict_errors.items():
    #     fig, ax1 = plt.subplots(figsize=(11, 8))
    #     ax1.hist(v, bins=30, range=(0, 60))
    #     ax1.set_title(k)
    #     ax1.set_xlabel("Errors in pixel")
    #     os.makedirs(os.path.join(args.prediction, "errors"), exist_ok=True)
    #     plt.savefig(os.path.join(args.prediction,
    #                 "errors", f"{k}_detection_error.png"))
    #     plt.close(fig)

    # print(errors)
    if not os.path.exists(os.path.join(args.prediction, "errors")):
        os.makedirs(os.path.join(args.prediction, "errors"), exist_ok=True)

    for i, frame_index in enumerate(frame_indices):
        results["file_errors"][frame_index] = {
            "recall": str(accuracies[i]),
            "mean_reproj_error": str(np.mean(list(errors[i].values()))),
            "reprojection_errors": errors[i]
        }

    with open(os.path.join(args.prediction, "errors", "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
