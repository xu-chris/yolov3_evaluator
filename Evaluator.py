from yolo import YOLO
from PIL import Image as PIL_Image
import cv2
import os
from BoundingBox import BoundingBox
from ErrorType import ErrorType


class Evaluator:

    def __init__(self, args, test_data_folder, test_data_annotation_file):
        self.args = args
        self.test_data_folder = test_data_folder
        self.test_data_annotation_file = test_data_annotation_file

    def run_tests(self):

        # Read likes from test file
        with open(self.test_set_annotation_path) as f:
            lines = f.readlines()

        # Make result folder
        folder_hash = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs('result', exist_ok=True)
        os.makedirs('result' + '/' + folder_hash, exist_ok=True)

        # Create empty classification result dict
        result = []

        # For each line
        yolo = YOLO(**args)
        for line in lines:
            # Read image via image path
            line = annotation_line.split()
            cv2_img = cv2.imread(line[0])
            image_shape = {
                'width': cv2_img.shape[1],
                'height': cv2_img.shape[0]
            }
            pil_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            color_image = PIL_Image.fromarray(pil_image)

            # Read box details
            boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

            # Run detection
            r_image, out_boxes, out_scores, out_classes = yolo.detect_image(color_image)

            # Compare each box
            for out_box, out_score, out_class in zip(out_boxes, out_scores, out_classes):

                # If no box is given anymore, this means we have too many detections. Exit loop by append NOT_GIVEN as error
                if len(boxes) <= 0:
                    result.append({
                        'class': out_class,
                        'error_type': ErrorType.NOT_GIVEN
                    })

                    continue

                best_box = None
                # Find comparable box by checking the intersection over union
                for box in boxes:
                    box_obj = BoundingBox.from_list(box, image_shape)
                    # Calculate intersection over union
                    # Select the box with highest IoU and pop it from boxes array
                    pass
                # If none given, set the error as background error
                if best_box is None:
                    result.append({
                        'class': out_class,
                        'error_type': ErrorType.BACKGROUND
                    })

                    continue

                # Is correctly classified?
                correct_classified = None

                # BB to image size ratio
                bb_to_image_size_ratio = -1

                # Compare bounding box center with Ground truth
                bb_center_deviation = -1

                # Compare width and height
                bb_width_deviation = -1
                bb_height_deviation = -1

                # Conclude size deviation
                bb_size_deviation = -1

                # Save into dict of classification results
                result.append({
                    'class': out_class,
                    'bb_size_to_image_ratio': bb_to_image_size_ratio,
                    'average_precision': average_precision,
                    'bb_center_deviation': bb_center_deviation,
                    'bb_width_deviation': bb_width_deviation,
                    'bb_height_deviation': bb_height_deviation,
                    'bb_size_deviation': bb_size_deviation,
                    'correct_classified': correct_classified,
                    'error_type': None
                })

            # If still boxes are given, this means theses weren't classified at all. Count them as NOT_DETECTED errors
            for i in range(len(boxes)):
                result.append({
                    'class': gt_class,
                    'error_type': ErrorType.NOT_DETECTED
                })

        print('Closing YOLO session...')
        yolo.close_session()

        # Calculate mean average precision per class

        # Calculate error distribution per class

        # Calculate deviation errors per class

        # Calculate total mean average precision

        # Calculate total error distribution

        # Calculate total deviation errors
