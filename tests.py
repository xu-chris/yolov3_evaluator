import unittest
from BoundingBox import BoundingBox


class TestBoundingBox(unittest.TestCase):

    def test_creation_from_values(self):
        """
        Tests the creation of a bounding box by it's values
        """

        min_x = 0
        rel_min_x = 0
        min_y = 0
        rel_min_y = 0
        max_x = 100
        rel_max_x = 1
        max_y = 50
        rel_max_y = 0.5
        image_shape = [100, 100, 3]  # height, width, dimensions
        bb_class = 0
        score = 0.75
        center = [50, 25]
        rel_center = [0.5, 0.25]
        width = 100
        height = 50
        rel_width = 1
        rel_height = 0.5

        bounding_box = BoundingBox(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            bb_class=bb_class,
            image_shape=image_shape,
            score=score
        )

        # Check for correct setting of variables
        self.assertEqual(bounding_box.minX, min_x)
        self.assertEqual(bounding_box.minY, min_y)
        self.assertEqual(bounding_box.maxX, max_x)
        self.assertEqual(bounding_box.maxY, max_y)
        self.assertEqual(bounding_box.bb_class, bb_class)
        self.assertEqual(bounding_box.score, score)
        # Check for calculated values
        self.assertEqual(bounding_box.width, width)
        self.assertEqual(bounding_box.height, height)
        self.assertEqual(bounding_box.rel_minX, rel_min_x)
        self.assertEqual(bounding_box.rel_maxX, rel_max_x)
        self.assertEqual(bounding_box.rel_minY, rel_min_y)
        self.assertEqual(bounding_box.rel_maxY, rel_max_y)
        self.assertEqual(bounding_box.rel_width, rel_width)
        self.assertEqual(bounding_box.rel_height, rel_height)
        self.assertEqual(bounding_box.center, center)
        self.assertEqual(bounding_box.rel_center, rel_center)

    def test_creation_from_list(self):
        bb_list = [390, 1, 444, 456, 1]
        image_shape = [1080, 1920, 3]
        width = 54
        height = 455
        center = [417, 228.5]

        bounding_box = BoundingBox.from_list(bb_list, image_shape)

        # Check for correct setting of variables
        self.assertEqual(bounding_box.minX, bb_list[0])
        self.assertEqual(bounding_box.minY, bb_list[1])
        self.assertEqual(bounding_box.maxX, bb_list[2])
        self.assertEqual(bounding_box.maxY, bb_list[3])
        self.assertEqual(bounding_box.bb_class, bb_list[4])
        # Check for calculated values
        self.assertEqual(bounding_box.width, width)
        self.assertEqual(bounding_box.height, height)
        self.assertEqual(bounding_box.center, center)

    def test_creation_from_detection(self):
        bb_list = [143.18617, 579.51013, 355.62076, 607.54224]
        bb_class = 1
        score = 0.9335858225822449
        image_shape = [1080, 1920, 3]

        width = 28.03210999999999
        height = 212.43459000000001
        center = [593.5261849999999, 249.403465]

        bounding_box = BoundingBox.from_detection(bb_list, bb_class, score, image_shape=image_shape)

        # Check for correct setting of variables
        self.assertEqual(bounding_box.minX, bb_list[1])
        self.assertEqual(bounding_box.minY, bb_list[0])
        self.assertEqual(bounding_box.maxX, bb_list[3])
        self.assertEqual(bounding_box.maxY, bb_list[2])
        self.assertEqual(bounding_box.bb_class, bb_class)
        # Check for calculated values
        self.assertEqual(width, bounding_box.width)
        self.assertEqual(height, bounding_box.height)
        self.assertEqual(center, bounding_box.center)


if __name__ == '__main__':
    unittest.main()
