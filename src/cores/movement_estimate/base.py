from skimage.measure import block_reduce
from abc import ABC, abstractmethod
import numpy as np
import cv2

from src.cores.base import StormObject, StormsMap
from src.utils import convert_polygons_to_contours

class BaseTREC(ABC):
    def __init__(self):
        pass

    def average_storm_movement(
            self, storm: StormObject, map_shape: tuple[int, int], grid_y: np.ndarray, grid_x: np.ndarray, vy: np.ndarray, vx: np.ndarray
        ) -> tuple[float, float]:
        """
        Aproximating the movement of the storm using the velocity field, given the movement grids and velocity components.
        Returns (dy, dx)
        """
        block_size = int(grid_y[0] * 2)

        contours = convert_polygons_to_contours([storm.contour])
        mask = np.zeros(map_shape, dtype=np.uint8)
        cv2.fillPoly(mask, contours, color=1)

        crop_mask = mask[0:block_size * len(grid_y), 0:block_size * len(grid_x)]

        block_mask = block_reduce(crop_mask, block_size=(block_size,block_size), func=np.sum)
        total = np.sum(block_mask) + 1e-8
        dy = np.sum(vy * block_mask) / total
        dx = np.sum(vx * block_mask) / total

        return (dy, dx)

    @abstractmethod
    def estimate_movement(
            self, prev_map: StormsMap, curr_map: StormsMap
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate a field-based motion map. Returns (grid_y, grid_x, vy, vx).
        """
        pass