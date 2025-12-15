import numpy as np
import cv2
from datetime import datetime
from shapely import Polygon
from skimage.measure import block_reduce

from src.utils import convert_polygons_to_contours
from src.cores.base import StormObject, StormsMap

class CentroidStorm(StormObject):
    centroid: np.ndarray

    def __init__(self, polygon: Polygon, centroid: tuple[float, float], id: str=""):
        super().__init__(contour=polygon, id=id)
        self.centroid = np.array(centroid)

    def retrieve_movement(self, grid_y: np.ndarray, grid_x: np.ndarray, vy: np.ndarray, 
                          vx: np.ndarray, block_size: int, img_shape: tuple[int, int]) -> tuple[float, float]:
        """
        Retrieve movement of the current storm by averaging the motion vectors inside the storm area. 
        Returns (dy, dx)
        """
        if len(grid_y) == 0 or len(grid_x) == 0 or vy.size == 0 or vx.size == 0:
            raise ValueError("Grid_y, grid_x, vy, and vx must all be non-empty.")

        # create mask of the storm
        contours = convert_polygons_to_contours([self.contour])
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.fillPoly(mask, contours, color=1)
        
        # crop the mask to fit the block grid
        crop_mask = mask[0:block_size * len(grid_y), 0:block_size * len(grid_x)]

        block_mask = block_reduce(crop_mask, block_size=(block_size,block_size), func=np.sum)
        total = np.sum(block_mask) + 1e-8
        dy = np.sum(vy * block_mask) / total
        dx = np.sum(vx * block_mask) / total

        return dy, dx

class DbzStormsMap(StormsMap):
    storms: list[CentroidStorm]
    dbz_map: np.ndarray

    def __init__(self, storms: list[CentroidStorm], time_frame: datetime, dbz_map: np.ndarray):
        """
        Beside 2 default attributes, also keep track of `dbz_map` for computin correlation.
        """
        super().__init__(storms, time_frame)
        self.dbz_map = dbz_map

    def _retrieve_movement(self, block: np.ndarray, search_region: np.ndarray) -> np.ndarray:
        block = block.astype(np.float32)
        search_region = search_region.astype(np.float32)
        result = cv2.matchTemplate(search_region, block, cv2.TM_CCOEFF_NORMED)
        return np.unravel_index(np.argmax(result), result.shape)

    def trec_estimate(self, other: "DbzStormsMap", **kargs) -> tuple[
            list[tuple[float, float]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ]:
        """
        Generate a correlation map which show the movement between the current to other storm map.
        """
        block_size = kargs.get("block_size", 16)
        stride = kargs.get("stride", 16)
        local_buffer = kargs.get("buffer", 50)      # search region = block + expanded by local_buffer

        dbz_map_1 = self.dbz_map
        dbz_map_2 = other.dbz_map
        H, W = dbz_map_2.shape

        ys = list(range(0, H-block_size+1, stride))     # ys: list[start_idx of H-axis]
        xs = list(range(0, W-block_size+1, stride))     # xs: list[start_idx of W-axis]

        vy = np.zeros(shape=(len(ys), len(xs)))         # vy: keep the y-value of movement at corresponding position
        vx = np.zeros_like(vy)                          # vx: keep the y-value of movement at corresponding position

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                block = dbz_map_1[y:y+block_size, x:x+block_size]
                if np.std(block) < 1e-3:    # case std is too small => continue
                    continue

                # otherwise: get the search region
                y_search_low, y_search_high = max(0,y-local_buffer), min(H,y + block_size + local_buffer)   # ensure the seach region is not overflow.
                x_search_low, x_search_high = max(0,x-local_buffer), min(W,x + block_size + local_buffer)

                search_region = dbz_map_2[y_search_low:y_search_high, x_search_low:x_search_high]
                dy, dx = self._retrieve_movement(block, search_region)

                y_best, x_best = y_search_low + dy, x_search_low + dx
                vy[i][j] = y_best - y
                vx[i][j] = x_best - x
        
        # Get the center of the block
        grid_y = np.array(ys) + block_size / 2
        grid_x = np.array(xs) + block_size / 2

        return [storm.retrieve_movement(
                    grid_y=grid_y, grid_x=grid_x, vy=vy, vx=vx, 
                    block_size=block_size, img_shape=dbz_map_1.shape) for storm in self.storms
                ], (grid_y, grid_x, vy, vx)