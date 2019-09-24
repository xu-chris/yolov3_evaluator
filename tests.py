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


if __name__ == '__main__':
    unittest.main()
