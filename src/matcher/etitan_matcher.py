from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from shapely.affinity import translate
from scipy.optimize import linear_sum_assignment

from src.cores.base import StormObject, StormsMap, UpdateType
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
    update_type: UpdateType = field(default=UpdateType.MATCHED)

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
        prev_storms = storms_map_1.storms
        curr_storms = storms_map_2.storms

        prev_num_storms = len(prev_storms)
        curr_num_storms = len(curr_storms)

        ## keep track of matched storms
        assignments: list[MatchedStormPair] = []
        prev_matched_set = set()
        curr_matched_set = set()
        
        # 1. matching using TREC forecasting + movement estimation
        grid_y, grid_x, vy, vx = estimate_trec_by_blocks(storms_map_1, storms_map_2, block_size=correlation_block_size, 
                                                     stride=correlation_block_size)
        curr_polygons = [storm.contour for storm in storms_map_2.storms]

        for i, prev_storm in enumerate(storms_map_1.storms):
            dy, dx = average_storm_movement(prev_storm, storms_map_1.dbz_map.shape[:2], grid_y, grid_x, vy, vx)
            pred_pol = translate(prev_storm.contour, xoff=dx, yoff=dy)

            matching_indices = [curr_order for curr_order, curr_pol in enumerate(curr_polygons) if \
                            area_overlapping_ratio(pred_pol, curr_pol, mode='avg') > matching_overlap_threshold]

            # check and update matched sets
            if len(matching_indices) > 0:
                prev_matched_set.add(i)
                for matching_idx in matching_indices:
                    curr_matched_set.add(int(matching_idx))

            assignments.extend([
                    MatchedStormPair(prev_storm_order=i, curr_storm_order=int(curr_order), 
                                        estimated_movement=np.array([dy, dx])
                                        # estimated_movement=
                                        # np.array([curr_storms[int(curr_order)].centroid[0] - prev_storms[i].centroid[0], curr_storms[int(curr_order)].centroid[1] - prev_storms[i].centroid[1]])
                                    ) 
                    for curr_order in matching_indices
                ])

        prev_matched = list(prev_matched_set)
        curr_matched = list(curr_matched_set)
        
        # # 2. perform 2nd matching for unmatched storms

        cost_matrix, displacement_matrix = self._construct_disparity_matrix(
                storm_lst1=prev_storms, storm_lst2=curr_storms
            )

        ## adjust the cost matrix
        dt = (storms_map_2.time_frame - storms_map_1.time_frame).seconds / 3600     # unit: hr

        ## Establish invalid matches based on dynamic velocity constraint
        maximum_displacement_matrix = np.zeros_like(displacement_matrix)

        for i in range(len(prev_storms)):
            maximum_displacement_matrix[i,:] = self.dynamic_max_velocity(prev_storms[i].contour.area) * dt
        for j in range(len(curr_storms)):
            maximum_displacement_matrix[:,j] = np.minimum(
                maximum_displacement_matrix[:,j],
                self.dynamic_max_velocity(curr_storms[j].contour.area) * dt
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

        ## in case 2: use centroid displacement as estimated movement
        assignments.extend([
            MatchedStormPair(
                prev_storm_order=prev_idx,
                curr_storm_order=curr_idx,
                estimated_movement=np.array([
                    curr_storms[curr_idx].centroid[1] - prev_storms[prev_idx].centroid[1],
                    curr_storms[curr_idx].centroid[0] - prev_storms[prev_idx].centroid[0]
                ])
            )
            for prev_idx, curr_idx in zip(row_ind, col_ind) if assignment_mask[prev_idx, curr_idx]
        ])

        # 3. Resolve split and merge assignments
        prev_mapping = {prev_idx: [] for prev_idx in range(prev_num_storms)}
        curr_mapping = {curr_idx: [] for curr_idx in range(curr_num_storms)}

        for match_id, match in enumerate(assignments):
            prev_mapping[match.prev_storm_order].append(match_id)
            curr_mapping[match.curr_storm_order].append(match_id)

        ## 3.1 split detection => one-to-many from prev to curr
        new_assignments = []
        for prev_idx, match_indices in prev_mapping.items():
            if len(match_indices) > 1:      # largest storm carries the track
                # find the largest storm
                best_id = match_indices[0]
                best_area = storms_map_2.storms[assignments[best_id].curr_storm_order].contour.area

                for match_id in match_indices[1:]:
                    curr_area = storms_map_2.storms[assignments[match_id].curr_storm_order].contour.area
                    if curr_area > best_area:
                        best_area = curr_area
                        best_id = match_id

                # update assignments SPLITTED
                for match_id in match_indices:
                    if match_id != best_id:
                        assignments[match_id].update_type = UpdateType.SPLITTED
        
        ## 3.2 merge detection => many-to-one from prev to curr
        for curr_idx, match_indices in curr_mapping.items():
            if len(match_indices) == 0:
                # new storm
                new_assignments.append(MatchedStormPair(
                    prev_storm_order=-1,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.NEW
                ))

            if len(match_indices) > 1:    # largest storm continues the track
                # find the largest storm
                best_id = match_indices[0]
                best_area = storms_map_1.storms[assignments[best_id].prev_storm_order].contour.area

                for match_id in match_indices[1:]:
                    prev_area = storms_map_1.storms[assignments[match_id].prev_storm_order].contour.area
                    if prev_area > best_area:
                        best_area = prev_area
                        best_id = match_id

                # update assignments MERGED
                for match_id in match_indices:
                    if match_id != best_id:
                        assignments[match_id].update_type = UpdateType.MERGED

        assignments.extend(new_assignments)
        return assignments