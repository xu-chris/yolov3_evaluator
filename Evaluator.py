from detector.yolo import YOLO
from PIL import Image as PIL_Image
import cv2
import os
from BoundingBox import BoundingBox
from ErrorType import ErrorType
from QueryResult import QueryResult
from datetime import datetime
import numpy as np
from progress.bar import ChargingBar
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DetectionResult:
    """
    Testifies the result of the single bounding box detection result
    """

    def __init__(self, bb_class=None, score=None, bb_to_image_size_ratio=None, best_box_iou=None, bb_center_deviation=None, bb_width_deviation=None, bb_height_deviation=None, bb_size_deviation=None, classification_result=None, detection_result=None, query_result=None, running_precision=None,
                 running_recall=None, average_precision=None):
        self.bb_class = bb_class
        self.score = score
        self.bb_to_image_size_ratio = bb_to_image_size_ratio
        self.best_box_iou = best_box_iou
        self.bb_center_deviation = bb_center_deviation
        self.bb_width_deviation = bb_width_deviation
        self.bb_height_deviation = bb_height_deviation
        self.bb_size_deviation = bb_size_deviation
        self.classification_result = classification_result
        self.detection_result = detection_result
        self.query_result = query_result
        self.running_precision = running_precision
        self.running_recall = running_recall
        self.average_precision = average_precision

    def to_list(self):
        return [
            self.bb_class,
            self.score,
            self.bb_to_image_size_ratio,
            self.best_box_iou,
            self.bb_center_deviation,
            self.bb_width_deviation,
            self.bb_height_deviation,
            self.bb_size_deviation,
            self.classification_result,
            self.detection_result,
            self.query_result,
            self.running_precision,
            self.running_recall,
            self.average_precision
        ]


class Evaluator:

    def __init__(self, args, test_data_folder, test_data_annotation_file, test_data_classes_file):
        self.args = args
        self.test_data_folder = test_data_folder
        self.test_data_annotation_file = test_data_annotation_file
        self.test_data_classes_file = test_data_classes_file

        self.bb_classes = None
        self.bb_test_results: [] = None
        self.test_results: [] = None
        self.test_results_df = None
        self.test_data_plan = []
        self.result_folder = None

        self.read_bb_classes()
        self.read_test_data()
        self.make_results_folder()

    def read_bb_classes(self):
        with open(self.test_data_classes_file) as f:
            bb_classes = f.readlines()

        self.bb_classes = [str(bb_class.rstrip('\n')) for bb_class in bb_classes]

    def map_bb_class_id_to_string(self, bb_class_id):
        return str(self.bb_classes[bb_class_id])

    def read_test_data(self):
        # Read likes from test file
        with open(self.test_data_annotation_file) as f:
            data_lines = f.readlines()

        for line in data_lines:
            # Read image via image path
            line = line.split()

            if not line[0] or not line[1]:
                continue

            boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

            self.test_data_plan.append({
                'image_path': line[0],
                'ground_truths': boxes
            })

    def make_results_folder(self):
        # Make result folder
        folder_hash = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs('result', exist_ok=True)
        os.makedirs('result' + '/' + folder_hash, exist_ok=True)
        self.result_folder = 'result' + '/' + folder_hash + '/'

    def run_detection(self, iou_threshold=0.5, epsilon=1e-5):

        process_images_bar = ChargingBar('Running detection with all images', max=len(self.test_data_plan))

        # Create empty classification result dict
        detection_results = []

        precision_recall = []

        for bb_class in self.bb_classes:
            precision_recall.append({
                'total_true_positives': 0,
                'total_false_positives': 0,
                'total_false_negatives': 0
            })

        # For each line
        yolo = YOLO(**self.args)
        for query in self.test_data_plan:

            cv2_img = cv2.imread(query['image_path'])
            image_shape = cv2_img.shape
            pil_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            color_image = PIL_Image.fromarray(pil_image)

            # Read box details
            boxes = query['ground_truths']
            # Run detection
            r_image, out_boxes, out_scores, out_classes = yolo.detect_image(color_image)

            # Compare each box
            for out_box, out_score, out_class in zip(out_boxes, out_scores, out_classes):

                detection_result = DetectionResult()
                detection_result.bb_class = out_class
                detection_result.score = out_score

                # Create the box
                out_box = BoundingBox.from_detection(out_box, out_class, out_score, image_shape)

                # If no box is given anymore, this means we have too many detections. Continue loop by append NOT_GIVEN as error
                # These are false positives
                if len(boxes) <= 0:
                    detection_result.detection_result = ErrorType.NOT_GIVEN
                    detection_result.query_result = QueryResult.FALSE_POSITIVE
                    detection_results.append(detection_result.to_list())
                    continue

                # Find comparable box by checking the intersection over union
                best_box = None
                best_box_iou = None
                box = None
                for box in boxes:
                    box_list = box
                    box_obj = BoundingBox.from_list(box_list, image_shape)
                    # Calculate intersection over union
                    iou = box_obj.get_intersection_with(out_box)
                    if best_box is None or best_box_iou < iou:
                        best_box = box_obj
                        best_box_iou = iou
                        box = box
                # Select the box with highest IoU and pop it from boxes array.
                # If similar boxes exist, these might be fits for other boxes or false positives.
                boxes = np.delete(boxes, np.where(boxes == box), axis=0)
                detection_result.best_box_iou = best_box_iou

                # If none given, set the error as background error
                # It's a false positive
                if best_box is None:
                    detection_result.detection_result = ErrorType.NOT_GIVEN
                    detection_result.query_result = QueryResult.FALSE_POSITIVE
                    detection_results.append(detection_result.to_list())
                    continue

                # Is correctly classified?
                # If not, this is neither false positive nor false negative. It's basically completely wrong.
                if out_box.bb_class != best_box.bb_class:
                    detection_result.classification_result = ErrorType.WRONG_CLASS

                # BB to image size ratio
                detection_result.bb_to_image_size_ratio = out_box.rel_area

                # Compare bounding box center with Ground truth
                center_deviation_array = np.abs(np.array(out_box.rel_center) - np.array(best_box.rel_center))
                detection_result.bb_center_deviation = center_deviation_array[0] * center_deviation_array[1]

                # Compare width and height
                detection_result.bb_width_deviation = np.abs(out_box.rel_width - best_box.rel_width)
                detection_result.bb_height_deviation = np.abs(out_box.rel_height - best_box.rel_height)

                # Conclude size deviation
                detection_result.bb_size_deviation = np.abs(out_box.rel_area - best_box.rel_area)

                # Check if IoU is over PASCAL2012 threshold of 0.5
                if detection_result.best_box_iou > iou_threshold:
                    detection_result.query_result = QueryResult.TRUE_POSITIVE
                else:
                    detection_result.query_result = QueryResult.FALSE_POSITIVE

                # Calculate precision and recall
                if detection_result.query_result == QueryResult.TRUE_POSITIVE:
                    precision_recall[out_class]['total_true_positives'] += 1

                elif detection_result.query_result == QueryResult.FALSE_POSITIVE:
                    precision_recall[out_class]['total_false_positives'] += 1

                detection_result.running_precision = precision_recall[out_class]['total_true_positives'] / (precision_recall[out_class]['total_true_positives'] + precision_recall[out_class]['total_false_positives'] + epsilon)
                detection_result.running_recall = precision_recall[out_class]['total_true_positives'] / (precision_recall[out_class]['total_true_positives'] + precision_recall[out_class]['total_false_negatives'] + epsilon)

                # Save into dict of classification results
                detection_results.append(detection_result.to_list())

            # If still boxes are given, this means theses weren't classified at all.
            # Count them as NOT_DETECTED errors, False negatives
            if len(boxes) != 0:
                for box in boxes:
                    box = BoundingBox.from_list(box, image_shape=image_shape)
                    detection_result = DetectionResult(
                        bb_class=box.bb_class,
                        detection_result=ErrorType.NOT_DETECTED,
                        query_result=QueryResult.FALSE_NEGATIVE
                    )

                    # Calculate precision and recall
                    precision_recall[box.bb_class]['total_false_negatives'] += 1

                    detection_result.running_precision = precision_recall[box.bb_class]['total_true_positives'] / (precision_recall[box.bb_class]['total_true_positives'] + precision_recall[box.bb_class]['total_false_positives'] + epsilon)
                    detection_result.running_recall = precision_recall[box.bb_class]['total_true_positives'] / (precision_recall[box.bb_class]['total_true_positives'] + precision_recall[box.bb_class]['total_false_negatives'] + epsilon)

                    detection_results.append(detection_result.to_list())

            process_images_bar.next()

        self.test_results = detection_results
        process_images_bar.finish()
        yolo.close_session()

    def convert_results_into_df(self):

        # Format results to a better format
        results_columns = [
            'bb_class',
            'score',
            'bb_to_image_size_ratio',
            'best_box_iou',
            'bb_center_deviation',
            'bb_width_deviation',
            'bb_height_deviation',
            'bb_size_deviation',
            'classification_result',
            'detection_result',
            'query_result',
            'running_precision',
            'running_recall',
            'average_precision'
        ]
        results_df = pd.DataFrame(self.test_results, columns=results_columns)
        results_df[['bb_class']] = results_df[['bb_class']].astype(int)
        results_df['bb_class_string'] = [self.map_bb_class_id_to_string(int(class_id)) for class_id in results_df['bb_class']]

        self.test_results_df = results_df

    def evaluate_results(self):

        self.convert_results_into_df()
        self.test_results_df.to_csv(self.result_folder + 'analysis_result.csv')

        evaluation_result = []

        evaluation_results_file = open(self.result_folder + "evaluation_results.md", "w+")
        evaluation_results_file.write('# Experimental results of given model on {}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))

        # Interpolate through 11 recall bins according to Pascal 2012
        for bb_class in self.bb_classes:
            evaluation_results_file.write('\n## Evaluation result for class {}\n'.format(bb_class))
            df = self.test_results_df[self.test_results_df.bb_class_string == bb_class]
            df_precision_recall = df[df.running_recall.notnull() & df.running_precision.notnull()]
            recall_bins = {0.0: 0.0, 0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0, 0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
            for precision, recall in zip(df_precision_recall.running_precision.values, df_precision_recall.running_recall.values):
                # Check if every bin has it's maximum value
                recall_bin = np.ceil(recall * 10) / 10
                recall_bins[recall_bin] = max(recall_bins[recall_bin], precision)

            # Check if right recall bins have a higher precision. If so, overwrite given bin
            last_value = 0.0
            for key in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:

                if recall_bins[key] < last_value:
                    recall_bins[key] = last_value
                else:
                    last_value = recall_bins[key]

            print('Recall bins for class {}: {}'.format(bb_class, recall_bins))

            evaluation_results_file.write('- Recall bins for class {}: {}\n'.format(bb_class, recall_bins))

            precisions = list(recall_bins.values())
            average_precision = float(np.sum(np.array(precisions))) * float(1/11)
            print('AP for class {}: {}'.format(bb_class, average_precision))
            evaluation_results_file.write('- AP for class {}: {}\n'.format(bb_class, average_precision))

            # Calculate error distribution per class
            df_class_score = df[df.score.notnull()]
            average_classification_score = np.sum(df_class_score.score.values) / len(df_class_score.score.values)
            print('Average classification score for class {}: {}'.format(bb_class, average_classification_score))
            evaluation_results_file.write('- Average classification score for class {}: {}\n'.format(bb_class, average_classification_score))

            # Make graphic about accumulated detection errors

            percentage_wrong_class = len(df[df.classification_result == ErrorType.WRONG_CLASS].values) / len(df.index)
            print('Classified wrongly for class {}: {}'.format(bb_class, percentage_wrong_class))
            evaluation_results_file.write('- Classified wrongly for class {}: {}\n'.format(bb_class, percentage_wrong_class))

            # Conclude result
            evaluation_result.append({
                'average_precision': average_precision,
                'average_classification_score': average_classification_score,
                'bb_class': bb_class
            })

        # Calculate total mean average precision
        mean_average_precision = 0.0
        for class_result in list(evaluation_result):
            mean_average_precision = mean_average_precision + class_result['average_precision'] / len(self.bb_classes)

        evaluation_results_file.write('\n# Overall result\n')
        print('Mean average precision: {}'.format(mean_average_precision))
        evaluation_results_file.write('- Mean average precision: {}\n'.format(mean_average_precision))

        # Calculate total error distribution
        with sns.plotting_context("paper"):
            sns.set(style="whitegrid")
            pal = sns.color_palette("husl", 8)

            # Draw a nested barplot to show survival for class and sex
            g = sns.countplot(x='bb_class_string', hue="detection_result", palette=pal, data=self.test_results_df[self.test_results_df.detection_result.notnull()])
            g.get_figure().savefig(self.result_folder + "detection_errors_per_class.svg")

            evaluation_results_file.write('![alt Detection errors per class](detection_errors_per_class.svg)\n')

            # Calculate total deviation errors
            plt.clf()
            sns.set()

            # Use cubehelix to get a custom sequential palette
            df_detection_deviations = self.test_results_df[['bb_class_string','bb_center_deviation','bb_width_deviation','bb_height_deviation','bb_size_deviation']]
            df_detection_deviations.dropna()

            # Show each distribution with both violins and points
            h = sns.catplot(col='bb_class_string', data=df_detection_deviations, palette=pal, inner="stick", kind="violin", bw=.2, height=4, aspect=1.5)

            h.savefig(self.result_folder + "bounding_box_deviations.svg")
            evaluation_results_file.write('![alt Bounding Box Deviations](bounding_box_deviations.svg)\n')

        evaluation_results_file.close()
