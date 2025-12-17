import numpy as np

from .base import BaseTREC
from .trec_estimate import TREC

class COTREC(BaseTREC):
    trec_estimator: BaseTREC

    def __init__(self, max_velocity: float=100, block_size: int=8, stride: int=8):
        super().__init__()
        self.trec_estimator = TREC(
            block_size=block_size,
            stride=stride,
            max_velocity=max_velocity
        )
    
    # check incorrect vectors
    def _preprocess_trec_vectors(self, vy, vx, angle_thresh_deg=25):
        """
        Preprocess raw TREC vectors by:
        1) Replacing zero-velocity vectors
        2) Replacing vectors deviating > angle_thresh_deg from neighbors

        vy, vx : raw TREC velocity components
        """

        vy_correct = vy.copy()
        vx_correct = vx.copy()

        ny, nx = vx.shape
        angle_thresh = np.deg2rad(angle_thresh_deg)

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):

                u0 = vx_correct[i, j]
                v0 = vy_correct[i, j]
                mag = np.hypot(u0, v0)

                # Collect neighbors (only adjacent 4)
                u_nb = np.array([vx_correct[i-1, j], vx_correct[i+1, j], vx_correct[i, j-1], vx_correct[i, j+1]])
                v_nb = np.array([vy_correct[i-1, j], vy_correct[i+1, j], vy_correct[i, j-1], vy_correct[i, j+1]])
                
                # Remove invalid neighbors
                valid = ~np.isnan(u_nb)
                u_nb = u_nb[valid]
                v_nb = v_nb[valid]

                if len(u_nb) == 0:
                    continue

                # Condition A: zero velocity
                replace = (mag == 0)

                # Condition B: direction deviation
                if not replace:
                    theta0 = np.arctan2(v0, u0)

                    theta_nb = np.arctan2(v_nb, u_nb)
                    mean_u = np.mean(np.cos(theta_nb))
                    mean_v = np.mean(np.sin(theta_nb))
                    theta_mean = np.arctan2(mean_v, mean_u)

                    # smallest angular difference
                    dtheta = np.arctan2(
                        np.sin(theta0 - theta_mean),
                        np.cos(theta0 - theta_mean)
                    )

                    if np.abs(dtheta) > angle_thresh:
                        replace = True

                # Replace vector
                if replace:
                    vx_correct[i, j] = np.mean(u_nb)
                    vy_correct[i, j] = np.mean(v_nb)

        return vy_correct, vx_correct


    def _impose_variational(
            self,
            grid_y: np.ndarray, 
            grid_x: np.ndarray, 
            vy: np.ndarray, 
            vx: np.ndarray,
            n_iter: int = 20,
            omega: float = 1.8  # relaxation factor for SOR
        ):
        """
        Implemetation of Continuous TREC smoothing algorithm.
        Returns smoothed (vy, vx, grid_y, grid_x)
        """
        ny, nx = vx.shape
        
        dy = grid_y[1] - grid_y[0]
        dx = grid_x[1] - grid_x[0]

        div_v0 = (
            (vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * dx) +
            (vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2 * dy)
        )       # drop borders

        lambda_ = np.zeros_like(vx)

        for _ in range(n_iter):
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    lambda_[i, j] = (
                        (1 - omega) * lambda_[i, j]
                        +
                        omega * 0.25 * (
                            lambda_[i+1, j] + lambda_[i-1, j] +
                            lambda_[i, j+1] + lambda_[i, j-1]
                            + 2 * (dx**2) * div_v0[i-1, j-1]
                        )
                    )


        vx_smooth = vx.copy()
        vy_smooth = vy.copy()

        vx_smooth[1:-1, 1:-1] = (
            0.25 * (
                vx[1:-1, 2:] + 2 * vx[1:-1, 1:-1] + vx[1:-1, :-2]
            )
            + 0.5 * (
                (lambda_[1:-1, 2:] - lambda_[1:-1, :-2]) / (2 * dx)
            )
        )

        vy_smooth[1:-1, 1:-1] = (
            0.25 * (
                vy[2:, 1:-1] + 2 * vy[1:-1, 1:-1] + vy[:-2, 1:-1]
            )
            + 0.5 * (
                (lambda_[2:, 1:-1] - lambda_[:-2, 1:-1]) / (2 * dy)
            )
        )

        return vy_smooth, vx_smooth

    def estimate_movement(self, prev_map, curr_map):
        # estimate raw TREC vectors
        grid_y, grid_x, vy, vx = self.trec_estimator.estimate_movement(prev_map, curr_map)

        # V1: correct raw TREC vectors
        vy_correct, vx_correct = self._preprocess_trec_vectors(vy, vx)

        # V2: smooth TREC vectors using variational approach
        vy_smooth, vx_smooth = self._impose_variational(grid_y, grid_x, vy_correct, vx_correct)

        return grid_y, grid_x, vy_smooth, vx_smooth