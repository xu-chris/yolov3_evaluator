# YOLO v3 Evaluator

So you might have started using the [YOLO v3 Trainer](https://github.com/creichel/yolov3_trainer) and used the [YOLO v3 detector](https://github.com/creichel/yolov3_detector) successfully, but you really don't know if the given model is objectively *really* good, right?

This repository is inspired by the error analysis and evaluation of [YOLO v1](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html), as well as the tool [Diagnosing Errors in Detectors](http://dhoiem.web.engr.illinois.edu/projects/detectionAnalysis/) by Derek Hoiem. It uses Python to calculate the errors in the following error categories:
- Mean Average Precision
- Error rate caused by displacement of the bounding box (checking the centers of the detected bounding box with the ground truth)
- Error rate caused by a wrong sizing of the bounding box (checking the width and height of the bounding box)
- Error rate caused by a wrong classification

This repository is using the [YOLO v3 Detector](https://github.com/creichel/yolov3_detector) as a dependency. Ideally, the evaluation script should be very easily modifiable so you could possibly use any detector (as long as you're getting valuable information back).

## Getting started

`TBD`