from enum import Enum


class ErrorType(Enum):
    NOT_GIVEN = 0
    NOT_DETECTED = 1
    WRONG_CLASS = 2
    FALSE_DETECTION = 3
