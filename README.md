# YOLO v3 Evaluator

So you might have started using the [YOLO v3 Trainer](https://github.com/creichel/yolov3_trainer) and used the [YOLO v3 detector](https://github.com/creichel/yolov3_detector) successfully, but you really don't know if the given model is objectively *really* good, right?

This repository is inspired by the error analysis and evaluation of [YOLO v1](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html), as well as the tool [Diagnosing Errors in Detectors](http://dhoiem.web.engr.illinois.edu/projects/detectionAnalysis/) by Derek Hoiem. 

It calculates a percentage-based error rate based on these errors:
- Detected object is not given in Ground Truth (also known as "Background classification")
- Given object in ground truth was not detected
- Wrong classification of object

Additionally, the following precision metrics were used to determine the precision problems that lead to a worser mAP:
- Mean Average Precision
- Error rate caused by displacement of the bounding box (checking the centers of the detected bounding box with the ground truth)
- Error rate caused by a wrong sizing of the bounding box (checking the width and height of the bounding box)

This repository is using the [YOLO v3 Detector](https://github.com/creichel/yolov3_detector) as a dependency. Ideally, the evaluation script should be very easily modifiable so you could possibly use any detector (as long as you're getting valuable information back).

## Getting started

1. Set up your test data by placing the images and annotations into `test_data`
2. Copy your model into `model` folder
3. Set up the detection arguments in `evaluation.py` script. Feel free to setup other parameters there as well.
4. Install dependencies by running `pip install -r requirements.txt`
5. Run `python evaluation.py` script

You will find some graphs and metrics in the `result` folder after the script has finished running the tests.