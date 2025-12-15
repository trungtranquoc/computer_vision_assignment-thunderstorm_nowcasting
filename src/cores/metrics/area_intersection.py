import cv2
import numpy as np
from src.cores.base import StormObject
from typing import Union
from shapely import Polygon

def area_intersection(storm1: StormObject, storm2: StormObject) -> float:
    """
        Calculate the area of intersection between two storm objects.

        Args:
            storm1 (StormObject): The first storm object.
            storm2 (StormObject): The second storm object.

        Returns:
            float: The area of intersection between the two storm objects.
    """
    # Create blank images to draw the contours
    img1 = np.zeros((1000, 1000), dtype=np.uint8)
    img2 = np.zeros((1000, 1000), dtype=np.uint8)

    # Draw the contours on the blank images
    cv2.drawContours(img1, [storm1.contour], -1, color=255, thickness=-1)
    cv2.drawContours(img2, [storm2.contour], -1, color=255, thickness=-1)

    # Calculate the intersection
    intersection = cv2.bitwise_and(img1, img2)

    # Calculate the area of the intersection
    area = cv2.countNonZero(intersection)

    return area

def area_overlapping_ratio(
        storm1: Union[StormObject, Polygon], storm2: Union[StormObject, Polygon], mode: str = 'avg'
    ) -> float:
    """
    Compute overlapping ratio between 2 storms.

    Args:
        storm1 (Union[StormObject, Polygon]): The first storm object or polygon.
        storm2 (Union[StormObject, Polygon]): The second storm object or polygon.
        mode (str, optional): The mode to compute the overlapping ratio. 
                              Options are 'left', 'right', 'min', 'max', 'avg'. Defaults to 'avg'.
    """
    polyg_1 = storm1.polygon if isinstance(storm1, StormObject) else storm1
    polyg_2 = storm2.polygon if isinstance(storm2, StormObject) else storm2

    if mode == 'avg':
        return (polyg_1.intersection(polyg_2).area / polyg_1.area + polyg_1.intersection(polyg_2).area / polyg_2.area) / 2
    elif mode == 'left':
        return polyg_1.intersection(polyg_2).area / polyg_1.area
    elif mode == 'right':
        return polyg_1.intersection(polyg_2).area / polyg_2.area
    elif mode == 'min':
        return polyg_1.intersection(polyg_2).area / max(polyg_1.area, polyg_2.area)
    elif mode == 'max':
        return polyg_1.intersection(polyg_2).area / min(polyg_1.area, polyg_2.area)
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'left', 'right', 'min', 'max', 'avg'.")