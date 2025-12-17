import numpy as np
import cv2

from src.cores.base import StormsMap
from .base import BaseTREC

class TREC(BaseTREC):
    block_size: int
    stride: int
    max_velocity: float

    def __init__(self, block_size: int = 8, stride: int = 8, max_velocity: float = 100):
        super().__init__()
        self.block_size = block_size
        self.stride = stride
        self.max_velocity = max_velocity

    def estimate_movement(
            self, prev_map: StormsMap, curr_map: StormsMap
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate a field-based motion map using TREC. Returns (grid_y, grid_x, vy, vx).
        """
        dbz_map_1 = prev_map.dbz_map
        dbz_map_2 = curr_map.dbz_map
        H, W = dbz_map_2.shape

        dt = (curr_map.time_frame - prev_map.time_frame).seconds / 3600   # scaled to hour
        local_buffer = int(self.max_velocity * dt)

        ys = list(range(0, H-self.block_size+1, self.stride))     # ys: list[start_idx of H-axis]
        xs = list(range(0, W-self.block_size+1, self.stride))     # xs: list[start_idx of W-axis]

        vy = np.zeros(shape=(len(ys), len(xs)))         # vy: keep the y-value of movement at corresponding position
        vx = np.zeros_like(vy)                          # vx: keep the y-value of movement at corresponding position

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                block = dbz_map_1[y:y+self.block_size, x:x+self.block_size]
                if np.std(block) < 1e-3:    # case std is too small => continue
                    continue

                # otherwise: get the search region
                y_search_low, y_search_high = max(0,y-local_buffer), min(H,y + self.block_size + local_buffer)   # ensure the seach region is not overflow.
                x_search_low, x_search_high = max(0,x-local_buffer), min(W,x + self.block_size + local_buffer)

                # Perform template matching
                search_region = dbz_map_2[y_search_low:y_search_high, x_search_low:x_search_high]
                res = cv2.matchTemplate(search_region.astype(np.float32), block.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                dy, dx = np.unravel_index(np.argmax(res), res.shape)
            
                # Get the best matching position
                y_best, x_best = y_search_low + dy, x_search_low + dx
                vy[i][j] = y_best - y
                vx[i][j] = x_best - x
        
        # Get the center of the block
        grid_y = np.array(ys) + self.block_size / 2
        grid_x = np.array(xs) + self.block_size / 2

        return grid_y, grid_x, vy, vx