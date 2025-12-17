import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point

from src.cores.base import StormsMap, StormObject
from src.cores.base import UpdateType
from .etitan_matcher import MatchedStormPair

MAX_VELOCITY = 500
class SimpleMatcher:
    max_velocity: float     # unit: pixel/hr

    def __init__(self, max_velocity: float = MAX_VELOCITY):
        self.max_velocity = max_velocity

    def _construct_disparity_matrix(
            self, storm_lst1: list[StormObject], storm_lst2: list[StormObject]
        ) -> tuple[np.ndarray, np.ndarray]:
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
            self, storm_map1: StormsMap, storm_map2: StormsMap
        ) -> np.ndarray:
        """
        Match storms between 2 time frame.

        Args:
            storm_map1 (StormsMap): storm map in the 1st frame.
            storm_map2 (StormsMap): storm map in the 2nd frame.
        
        Returns:
            assignments (np.ndarray): Array of (prev_idx, curr_idx) pairs representing matched storms.
        """
        assignments: list[MatchedStormPair] = []

        # Handle no storm found cases
        if len(storm_map2.storms) == 0:
            return []
        if len(storm_map1.storms) == 0:
            return [MatchedStormPair(
                prev_storm_order=-1,
                curr_storm_order=curr_idx,
                update_type=UpdateType.NEW
            ) for curr_idx in range(len(storm_map2.storms))]

        dt = (storm_map2.time_frame - storm_map1.time_frame).seconds / 3600     # unit: hr
        max_displacement = dt * self.max_velocity

        cost_matrix, displacement_matrix = self._construct_disparity_matrix(storm_map1.storms, storm_map2.storms)
        invalid_mask = displacement_matrix > max_displacement

        cost_matrix = cost_matrix + invalid_mask.astype(np.float64) * 2000      # add penalty to those violated
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignment_mask = np.zeros_like(invalid_mask, dtype=bool)
        assignment_mask[row_ind, col_ind] = True

        idx_pairs = np.argwhere(assignment_mask & np.logical_not(invalid_mask))

        # Update the history movement
        for prev_idx, curr_idx in idx_pairs:
            assignments.append(MatchedStormPair(
                prev_storm_order=prev_idx,
                curr_storm_order=curr_idx,
                update_type=UpdateType.MATCHED,
                estimated_movement=np.array([
                    storm_map2.storms[curr_idx].centroid[1] - storm_map1.storms[prev_idx].centroid[1],
                    storm_map2.storms[curr_idx].centroid[0] - storm_map1.storms[prev_idx].centroid[0]
                ])
            ))

        # resolve merge & split
        ## mapping: dict where key -> index of storm; value -> list of tuple[storm_id]
        mapping_prev = {int(prev_idx): [int(curr_idx)] for prev_idx, curr_idx in idx_pairs}
        mapping_curr = {int(curr_idx): [int(prev_idx)] for prev_idx, curr_idx in idx_pairs}

        prev_assigned, curr_assigned = mapping_prev.keys(), mapping_curr.keys()
        prev_unassigned = [idx for idx in range(len(storm_map1.storms)) if idx not in prev_assigned]
        curr_unassigned = [idx for idx in range(len(storm_map2.storms)) if idx not in curr_assigned]

        if len(prev_unassigned) > 0 or len(curr_unassigned) > 0:  # if any unassigned => resolve
            pred_storms_map = StormsMap([
                    storm.forecast(dt=dt)
                    for storm in storm_map1.storms
                ], time_frame=storm_map2.time_frame)

            # Check for merging
            for prev_idx in prev_unassigned:
                pred_storm = pred_storms_map.storms[prev_idx]

                # Find storms that the predicted centroid fall into.
                candidates = [idx for idx, storm in enumerate(storm_map2.storms) if storm.contour.contains(Point(pred_storm.centroid))]
                
                # Case: more than 1 candidates => choose one with maximum overlapping on prev_storm
                if len(candidates) > 1:
                    compute_overlapping = lambda pol: pred_storm.contour.intersection(pol).area / pred_storm.contour.area
                    max_idx = np.argmax([compute_overlapping(storm_map2.storms[j].contour) for j in candidates])
                    candidates = [candidates[max_idx]]
                
                mapping_prev[prev_idx] = candidates
                for cand_idx in candidates:
                    if cand_idx not in mapping_curr:
                        mapping_curr[cand_idx] = []
                    mapping_curr[cand_idx].append(prev_idx)
                    assignments.append(MatchedStormPair(
                        prev_storm_order=prev_idx,
                        curr_storm_order=cand_idx,
                        update_type=UpdateType.MERGED,
                        estimated_movement=np.array([
                            pred_storms_map.storms[cand_idx].centroid[1] - storm_map1.storms[prev_idx].centroid[1],
                            pred_storms_map.storms[cand_idx].centroid[0] - storm_map1.storms[prev_idx].centroid[0]
                        ])
                    ))

                # In MERGING case, history movement already updated in the previous matching step, so no new update here.

            # Check for splitting
            for curr_idx in curr_unassigned:
                curr_storm = storm_map2.storms[curr_idx]
                # Find predicted storms that the current centroid fall into.
                candidates = [
                        idx for idx, storm in enumerate(pred_storms_map.storms) \
                            if storm.contour.contains(Point(curr_storm.centroid))
                    ]
                
                # Case: more than 1 candidates => choose one with maximum overlapping on prev_storm
                if len(candidates) > 1:
                    max_idx = np.argmax([curr_storm.contour.intersection(pred_storms_map.storms[i].contour) / curr_storm.contour.area for i in candidates])
                    candidates = [candidates[max_idx]]
                
                mapping_curr[curr_idx] = [cand_idx for cand_idx in candidates]
                for cand_idx in candidates:
                    if cand_idx not in mapping_prev:
                        mapping_prev[cand_idx] = []
                    mapping_prev[cand_idx].append(curr_idx)

                    # In case of SPLITTING, simply update the history movement the same as the forecasting movement
                    assignments.append(MatchedStormPair(
                        prev_storm_order=cand_idx,
                        curr_storm_order=curr_idx,
                        update_type=UpdateType.SPLITTED,
                        estimated_movement=np.array([
                            pred_storms_map.storms[cand_idx].centroid[1] - storm_map1.storms[prev_idx].centroid[1],
                            pred_storms_map.storms[cand_idx].centroid[0] - storm_map1.storms[prev_idx].centroid[0]
                        ])
                    ))

        for curr_idx in range(len(storm_map2.storms)):
            if mapping_curr.get(curr_idx, []) == []:
                assignments.append(MatchedStormPair(
                    prev_storm_order=-1,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.NEW
                ))

        # Sort to ensure MATCHED are processed first
        assignments.sort(key=lambda x: x.update_type.value)
        return assignments