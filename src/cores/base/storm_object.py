from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union
import numpy as np
import warnings
from shapely.affinity import translate

from shapely.geometry import Polygon
from src.utils import convert_contours_to_polygons

@dataclass
class StormObject:
    contour: Polygon    # List of points represent contours
    history_movements: list[tuple[float, float]]
    centroid: tuple[float, float]

    contour_color: tuple[int, int, int]
    id: str    # Unique ID of this object for tracking over time
    original_id: str

    def __init__(self, 
                 contour: Union[Polygon, np.ndarray], 
                 centroid: Optional[tuple[float, float]] = None, 
                 id: str = "",
                 history_movements: Optional[list[tuple[float, float]]] = [],
                 contour_color: Optional[list[int]] = None
                ):
        if type(contour) is np.ndarray:
            contour = convert_contours_to_polygons([contour])[0]

        self.contour = contour
        self.id = id
        self.original_id = id
        self.history_movements = history_movements
        if contour_color is not None:
            self.contour_color = tuple(contour_color)
        else:
            self.contour_color = tuple(np.random.randint(0, 255, size=3).tolist())

        if centroid is None:            # No centroid specified ==> Simply use the geomtric centroid of contour boundary
            self.centroid = (contour.centroid.x, contour.centroid.y)
        else:
            self.centroid = centroid

    def copy(self) -> 'StormObject':
        return StormObject(
            contour=self.contour,
            id=self.id,
            history_movements=self.history_movements.copy()
        )
    
    def make_move(self, displacement: tuple[float, float]):
        """
        Move the storm by the given displacement.

        Args:
            displacement (tuple[float, float]): (dy, dx) displacement to move the storm.
        """
        dy, dx = displacement
        self.contour = translate(self.contour, xoff=dx, yoff=dy)
        self.centroid = (self.centroid[0] + dy, self.centroid[1] + dx)

    def clear_history(self):
        self.history_movements = []

    def track_history(self, previous_storm: "StormObject", optimal_movement: tuple[float, float] = (0, 0)):
        # Transfer historical movements from previous storm
        self.history_movements = previous_storm.history_movements.copy()
        self.contour_color = previous_storm.contour_color
        
        self.history_movements.append(optimal_movement)

    
    ###### Forecasting part ######
    def _interpolate_velocity(self, velocity_lst: list[np.ndarray], alpha_decay: float = 0.5):
        """
        Interpolate the velocity using weighted average with decay factor alpha_decay.

        Args:
            velocity_lst (list[np.ndarray]): list of velocity vectors.
            alpha_decay (float, default=0.5): the decay factor.

        Returns:
            interpolated_velocity (np.ndarray): the interpolated velocity.
        """
        if len(velocity_lst) == 1:
            return velocity_lst[0]
        
        weights = np.array([alpha_decay**i for i in range(len(velocity_lst))])
        total_w = np.sum(weights)
        return np.sum([displ * w / total_w for displ, w in zip(velocity_lst[::-1], weights)], axis=0)
    
    def get_movement(self) -> list[np.ndarray]:
        """
        Get the list of recorded movements for the storm.

        Args:
            storm_id (int): id of the storm.

        Returns:
            movement_lst (list[np.ndarray]): list of recorded movements.
        """
        history_movements = self.history_movements
        if len(history_movements) == 0:
            warnings.warn(f"No recorded movement for storm {self.id}. Returning None.")
            return None
        
        return self._interpolate_velocity(self.history_movements)
    
    def forecast(self, dt: float, default_motion: np.ndarray = np.array([0,0])) -> "StormObject":
        """
        Make a forecast for the next position in dt hours.
        
        Args:
            dt (float): the interval between the current and next frame.
            default_motion (np.ndarray, default): default motion used in case there is no recorded history.
        
        Returns:
            storm (CentroidStorm): the estimated storm in the next frame.
        """
        movement = self.get_movement()
        displacement = movement * dt if movement is not None else default_motion * dt
        
        new_storm = self.copy()
        new_storm.make_move(displacement)

        return new_storm