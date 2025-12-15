from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from shapely.affinity import translate
from scipy.optimize import linear_sum_assignment

from src.cores.base import StormObject, StormsMap
from src.cores.metrics import area_overlapping_ratio
from src.cores.movement_estimate import estimate_trec_by_blocks, average_storm_movement

@dataclass
class MatchedStormPair:
    """
    Maintain matched storms pairs information.
    """
    prev_storm_order: int
    curr_storm_order: int
    estimated_movement: np.ndarray = field(default=None)   # (dy, dx)

    def derive_motion_vector(self, dt: float) -> np.ndarray:
        return self.estimated_movement / dt
    

class EtitanMatcher:
    dynamic_max_velocity: Callable[[float], float]      # dynamic constraint for maximum velocity

    def __init__(self, dynamic_max_velocity: Callable[[float], float]):
        self.dynamic_max_velocity = dynamic_max_velocity
    
    def _construct_disparity_matrix(
            self, storm_lst1: list[StormObject], storm_lst2: list[StormObject]
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct the cost matrix for Hungarian matching.
        """
        # get square root of area difference
        area_lst1 = np.array([storm.contour.area for storm in storm_lst1])
        area_lst2 = np.array([storm.contour.area for storm in storm_lst2])
        area_matrix = np.sqrt(np.abs(area_lst1[:, None] - area_lst2[None, :]))

        # get centroid displacement
        centroid_lst1 = np.array([storm.centroid for storm in storm_lst1])
        centroid_lst2 = np.array([storm.centroid for storm in storm_lst2])
        centroid_displacement_matrix = np.linalg.norm(centroid_lst1[:,None,:] - centroid_lst2[None,:,:], axis=2)
        
        return area_matrix + centroid_displacement_matrix, centroid_displacement_matrix
    
    
    def match_storms(
            self, storms_map_1: StormsMap, storms_map_2: StormsMap, 
            correlation_block_size: int = 16, matching_overlap_threshold: float = 0.5
        ) -> list[MatchedStormPair]:
        """
        Match storms between 2 time frame.

        Args:
            storms_map_1 (StormsMap): storm map in the 1st frame.
            storms_map_2 (StormsMap): storm map in the 2nd frame.
        
        Returns:
            assignments (np.ndarray): Array of (prev_idx, curr_idx) pairs representing matched storms.
        """
        prev_num_storms = len(storms_map_1.storms)
        curr_num_storms = len(storms_map_2.storms)

        # keep track of matched storms
        assignments: list[MatchedStormPair] = []
        prev_matched_set = set()
        curr_matched_set = set()
        
        # Step 1: matching using Hungarian matching.
        grid_y, grid_x, vy, vx = estimate_trec_by_blocks(storms_map_1, storms_map_2, block_size=correlation_block_size, 
                                                     stride=correlation_block_size)
        curr_polygons = [storm.contour for storm in storms_map_2.storms]

        for i, prev_storm in enumerate(storms_map_1.storms):
            dy, dx = average_storm_movement(prev_storm, storms_map_1.dbz_map.shape[:2], grid_y, grid_x, vy, vx)
            pred_pol = translate(prev_storm.contour, xoff=dx, yoff=dy)
            
            scores = [area_overlapping_ratio(pred_pol, curr_pol, mode='avg') for curr_pol in curr_polygons]
            matching_indices = np.argwhere(np.array(scores) > matching_overlap_threshold)
            if len(matching_indices) > 0:
                prev_matched_set.add(i)
                for matching_idx in matching_indices:
                    curr_matched_set.add(int(matching_idx[0]))

            assignments.extend([(i, matching_idx[0]) for matching_idx in matching_indices])

        prev_matched = list(prev_matched_set)
        curr_matched = list(curr_matched_set)

        assignments = np.array(assignments)

        ## case: all storms are matched, or 1 side is fully matched
        if len(prev_matched) == prev_num_storms or len(curr_matched) == curr_num_storms:
            return assignments
        
        # Step 2: perform 2nd matching for unmatched storms 
        cost_matrix, displacement_matrix = self._construct_disparity_matrix(storms_map_1.storms, storms_map_2.storms)

        ## adjust the cost matrix
        dt = (storms_map_2.time_frame - storms_map_1.time_frame).seconds / 3600     # unit: hr

        # Establish invalid matches based on dynamic velocity constraint
        maximum_displacement_matrix = np.zeros_like(displacement_matrix)
        for i in range(prev_num_storms):
            maximum_displacement_matrix[i,:] = self.dynamic_max_velocity(storms_map_1.storms[i].contour.area) * dt
        for j in range(curr_num_storms):
            maximum_displacement_matrix[:,j] = np.minimum(
                maximum_displacement_matrix[:,j],
                self.dynamic_max_velocity(storms_map_2.storms[j].contour.area) * dt
            )

        # Construct invalid mask
        invalid_mask = displacement_matrix > maximum_displacement_matrix
        invalid_mask[prev_matched,:] = True
        invalid_mask[:,curr_matched] = True

        cost_matrix = np.where(invalid_mask, 1e6, cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignment_mask = np.zeros_like(cost_matrix, dtype=bool)
        assignment_mask[row_ind, col_ind] = True
        assignment_mask = np.where(invalid_mask, False, assignment_mask)

        assignments_2 = np.argwhere(assignment_mask)

        if len(assignments_2) == 0:
            return assignments
        if len(assignments) == 0:
            return assignments_2

        return np.concatenate([assignments, assignments_2], axis=0)