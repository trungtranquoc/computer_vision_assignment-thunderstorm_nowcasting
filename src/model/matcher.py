from typing import Callable
from shapely.affinity import translate
import numpy as np

from src.cores.metrics import area_overlapping_ratio
from .storm import CentroidStorm, DbzStormsMap

class Matcher():
    dynamic_max_velocity: Callable[[float], float]      # dynamic constraint for maximum velocity

    def __init__(self, dynamic_max_velocity: Callable[[float], float]):
        self.dynamic_max_velocity = dynamic_max_velocity
    
    def _construct_disparity_matrix(
            self, storm_lst1: list[CentroidStorm], storm_lst2: list[CentroidStorm]
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
            self, storms_map_1: DbzStormsMap, storms_map_2: DbzStormsMap, 
            correlation_block_size: int = 16, matching_overlap_threshold: float = 0.5
        ) -> np.ndarray:
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
        
        # Step 1: matching using Hungarian matching.
        prev_movements, _ = storms_map_1.trec_estimate(storms_map_2, block_size=correlation_block_size, stride=correlation_block_size)
        curr_polygons = [storm.contour for storm in storms_map_2.storms]

        prev_matched_set = set()
        curr_matched_set = set()

        assignments = []
        for i, (prev_storm, (dy, dx)) in enumerate(zip(storms_map_1.storms, prev_movements)):
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
        invalid_mask = np.ones_like(cost_matrix, dtype=bool)

        for i in range(prev_num_storms):
            # case: storm i in prev is already matched => continue
            if i in prev_matched:
                continue
            for j in range(curr_num_storms):
                # case: storm j in curr is already matched => continue
                if j in curr_matched:
                    continue

                # check for dynamic velocity constraint
                max_area = max(storms_map_1.storms[i].contour.area, storms_map_2.storms[j].contour.area)
                max_displacement = self.dynamic_max_velocity(max_area) * dt
                if displacement_matrix[i][j] <= max_displacement:
                    invalid_mask[i][j] = 0

        cost_matrix = cost_matrix + invalid_mask * 1e6
        row_ind, col_ind = self._hungarian_matching(cost_matrix)

        assignment_mask = np.zeros_like(cost_matrix, dtype=bool)
        assignment_mask[row_ind, col_ind] = True

        assignments_2 = np.argwhere(assignment_mask & np.logical_not(invalid_mask))

        return np.concatenate([assignments, assignments_2], axis=0)