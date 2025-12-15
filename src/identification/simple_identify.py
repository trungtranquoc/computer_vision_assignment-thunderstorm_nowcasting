import numpy as np
import cv2
from typing import List, Tuple
import numpy as np
from .base import BaseStormIdentifier

class SimpleContourIdentifier(BaseStormIdentifier):
    """
        Detect storm objects solely based on the contiguous spatial areas of pixels exceeding specified dBZ thresholds. 
    """
    def identify_storm(self, dbz_map: np.ndarray, threshold: int = 20, filter_area: int = 50) -> list[np.ndarray]:
        """
            Draw the DBZ contour for the image.

            Args:
                img: source image.

            Returns:
                List[np.ndarray]: A list of detected contours, each represented as an array of points.
        """
        if isinstance(dbz_map, np.ma.MaskedArray):
            dbz_map = dbz_map.filled(0)
        # Get the region
        region = (dbz_map >= threshold).astype(np.uint8)

        # Draw the contour
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [polygon for polygon in contours if cv2.contourArea(polygon) >= filter_area]
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        return contours