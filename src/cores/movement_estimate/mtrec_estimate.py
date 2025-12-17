import numpy as np
import cv2

from src.cores.base import StormsMap
from .base import BaseTREC

class MTREC(BaseTREC):
    max_velocity: float
    large_box_size: int
    small_box_size: int

    def __init__(self, max_velocity: float=100, large_box_size: int=16, small_box_size: int=8):
        self.max_velocity = max_velocity
        self.large_box_size = large_box_size
        self.small_box_size = small_box_size
    
    def _estimate_trec_by_blocks(self, dbz_map_1: np.ndarray, dbz_map_2: np.ndarray, block_size, stride, local_buffer):
        """
        Use TREC to estimate the velocity field between 2 frames. This is used as the first guess for storm matching.
        """
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

                vy[i, j] = dy + (y_search_low - y)
                vx[i, j] = dx + (x_search_low - x)

        # Get the center of the block
        grid_y = np.array(ys) + block_size / 2
        grid_x = np.array(xs) + block_size / 2

        return grid_y, grid_x, vy, vx
    
    def _fill_map(self, H, W, grid_y, grid_x, vy, vx, block_size):
        vy_full = np.zeros((H, W))
        vx_full = np.zeros((H, W))

        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                y_start = int(max(0, y - block_size / 2))
                y_end = int(min(H, y + block_size / 2))
                x_start = int(max(0, x - block_size / 2))
                x_end = int(min(W, x + block_size / 2))

                vy_full[y_start:y_end, x_start:x_end] = vy[i, j]
                vx_full[y_start:y_end, x_start:x_end] = vx[i, j]

        # Smooth the field
        vx_full = cv2.GaussianBlur(vx_full, (11, 11), 0)
        vy_full = cv2.GaussianBlur(vy_full, (11, 11), 0)

        return vy_full, vx_full

    def _interpolate(self, field):
        field = cv2.GaussianBlur(field, (31, 31), 0)
        return field

    def _truncate(self, vy, vx, max_displacement):
        mag = np.sqrt(vy**2 + vx**2)
        mask = mag > max_displacement
        vy[mask] = (vy[mask] / mag[mask]) * max_displacement
        vx[mask] = (vx[mask] / mag[mask]) * max_displacement
        
        return vy, vx

    def _advect(self, dbz_map, vy_full, vx_full):
        H, W = dbz_map.shape
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        Xd = X - vx_full
        Yd = Y - vy_full

        Xd = np.clip(Xd, 0, W - 1)
        Yd = np.clip(Yd, 0, H - 1)

        return cv2.remap(
            dbz_map,
            Xd.astype(np.float32),
            Yd.astype(np.float32),
            interpolation=cv2.INTER_LINEAR
        )
    
    def estimate_movement(
            self, prev_map: StormsMap, curr_map: StormsMap
        ):
        """
        Estimate a field-based motion map using MTREC. MTREC uses 2 scales of TREC and semi-Lagrangian advection.
        Returns (grid_y, grid_x, vy, vx).
        """
        dbz_map_1 = prev_map.dbz_map
        dbz_map_2 = curr_map.dbz_map
        H, W = dbz_map_2.shape

        dt = (curr_map.time_frame - prev_map.time_frame).seconds / 3600   # scaled to hour
        local_buffer = int(self.max_velocity * dt)
        
        # 1. Large-scale (Systematic) TREC
        grid_y_l, grid_x_l, vy_l, vx_l = self._estimate_trec_by_blocks(
                dbz_map_1, dbz_map_2, block_size=self.large_box_size, stride=self.large_box_size, local_buffer=local_buffer
            )
        
        # 2. Fill the velocity field. Vectors inside each block are the same.
        vy_l_full, vx_l_full = self._fill_map(H, W, grid_y_l, grid_x_l, vy_l, vx_l, block_size=self.large_box_size)
        
        # 3. Semi-Lagrangian Advection
        backward_advected_dbz = self._advect(dbz_map_2, vy_full=vy_l_full, vx_full=vx_l_full)

        # 4. Small-scale (Turbulent) TREC
        ## search for a small buffer since large-scale motion has been removed
        small_buffer = max(2, local_buffer // 3)
        grid_y_s, grid_x_s, vy_s, vx_s = self._estimate_trec_by_blocks(
                dbz_map_1, backward_advected_dbz, block_size=self.small_box_size, stride=self.small_box_size, local_buffer=small_buffer
            )
        
        # 5. Generate TREC field
        grid_y_s = grid_y_s.astype(np.int16)
        grid_x_s = grid_x_s.astype(np.int16)

        Y_s, X_s = np.meshgrid(grid_y_s, grid_x_s, indexing="ij")

        # 5. Generate TREC field
        vy_l_at_s = vy_l_full[Y_s, X_s]
        vx_l_at_s = vx_l_full[Y_s, X_s]

        vy_final = vy_s + vy_l_at_s
        vx_final = vx_s + vx_l_at_s

        # 6. Truncate and smooth final field
        vy_final, vx_final = self._truncate(vy_final, vx_final, self.max_velocity * dt)

        return grid_y_s, grid_x_s, vy_final, vx_final