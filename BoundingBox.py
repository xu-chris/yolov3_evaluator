class BoundingBox:

    def __init__(self, min_x, min_y, max_x, max_y, bb_class, image_shape, score=1):
        self.minX = min_x
        self.minY = min_y
        self.maxX = max_x
        self.maxY = max_y
        self.bb_class = bb_class
        self.image_shape = image_shape
        self.score = score

        self.rel_minX = None
        self.rel_minY = None
        self.rel_maxX = None
        self.rel_maxY = None
        self.width = None
        self.height = None
        self.rel_width = None
        self.rel_height = None
        self.center = None
        self.rel_center = None

        self.calculate_rel_positions()
        self.calculate_sizes()
        self.calculate_center()

    @classmethod
    def from_list(cls, box_as_list, image_shape) -> 'BoundingBox':
        return cls(min_x=box_as_list[0], min_y=box_as_list[1], max_x=box_as_list[2], max_y=box_as_list[3], bb_class=box_as_list[4], image_shape=image_shape)

    @classmethod
    def from_detection(cls, box_as_list, bb_class, score, image_shape) -> 'BoundingBox':
        return cls(min_x=box_as_list[0], min_y=box_as_list[1], max_x=box_as_list[2], max_y=box_as_list[3], bb_class=bb_class, image_shape=image_shape, score=score)

    def calculate_rel_positions(self):
        self.rel_minX = self.minX / self.image_shape[1]
        self.rel_maxX = self.maxX / self.image_shape[1]

        self.rel_minY = self.minY / self.image_shape[0]
        self.rel_maxY = self.maxY / self.image_shape[0]

    def calculate_sizes(self):
        self.width = self.maxX - self.minX
        self.height = self.maxY - self.minY

        self.rel_width = self.width / self.image_shape[1]
        self.rel_height = self.height / self.image_shape[0]

    def calculate_center(self):
        self.center = [
            self.minX + (self.maxX / 2),
            self.minY + (self.maxY / 2)
        ]

        self.rel_center = [
            self.center[0] / self.image_shape[1],
            self.center[1] / self.image_shape[0]
        ]