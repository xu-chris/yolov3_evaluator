from Evaluator import Evaluator

args = {
    "model_path": 'model/weights.h5',
    "anchors_path": 'model/anchors.txt',
    "classes_path": 'model/classes.txt',
    "score": 0.7,
    "iou": 0.15,
    "model_image_size": (608, 608),
    "gpu_num": 1,
}

if __name__ == '__main__':
    evaluator = Evaluator(args, 'test_data', 'test_data/test.txt')
    evaluator.run_tests()
