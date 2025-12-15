import numpy as np
import cv2
from skimage.measure import block_reduce

from src.cores.base import StormObject, StormsMap
from src.utils import convert_polygons_to_contours

def average_storm_movement(storm: StormObject, map_shape: tuple[int, int], grid_y: np.ndarray, grid_x: np.ndarray, vy: np.ndarray, vx: np.ndarray) -> tuple[float, float]:
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

def estimate_trec_by_blocks(prev_map: StormsMap, curr_map: StormsMap, 
                       block_size: int=16, stride: int=16, local_buffer: int=50):
        """
        Use TREC to estimate the velocity field between 2 frames. This is used as the first guess for storm matching.
        """
        dbz_map_1 = prev_map.dbz_map
        dbz_map_2 = curr_map.dbz_map
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

                # Perform template matching
                search_region = dbz_map_2[y_search_low:y_search_high, x_search_low:x_search_high]
                res = cv2.matchTemplate(search_region.astype(np.float32), block.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                dy, dx = np.unravel_index(np.argmax(res), res.shape)
            
                # Get the best matching position
                y_best, x_best = y_search_low + dy, x_search_low + dx
                vy[i][j] = y_best - y
                vx[i][j] = x_best - x
        
        # Get the center of the block
        grid_y = np.array(ys) + block_size / 2
        grid_x = np.array(xs) + block_size / 2

        return grid_y, grid_x, vy, vx