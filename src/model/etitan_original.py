import cv2
import numpy as np
from datetime import datetime
from shapely.geometry import Point, Polygon

from src.cores.base import StormsMap, StormObject, TrackingHistory
from src.identification import MorphContourIdentifier
from src.matcher import EtitanMatcher
from src.utils import convert_contours_to_polygons, convert_polygons_to_contours

class ETitanPrecipitationModel:
    """
    ETitan model implementation for thunderstorm nowcasting.
    """
    identifier: MorphContourIdentifier
    storms_maps: list[StormsMap]
    tracker: TrackingHistory
    matcher: EtitanMatcher

    def __init__(self, identifier: MorphContourIdentifier):
        self.identifier = identifier
        self.storms_maps = []
        self.matcher = EtitanMatcher(self._dynamic_max_velocity)
        self.tracker = None

    def _dynamic_max_velocity(self, area: float) -> float:
        """
        Dynamic constraint for maximum velocity based on storm area. The unit of velocity is pixel/hr.
        """
        if area < 300:
            return 500
        elif area < 500:
            return 750
        else:
            return 1000

    def identify_storms(self, dbz_map: np.ndarray, time_frame: datetime, map_id: str, threshold: float, filter_area: float) -> StormsMap:
        """
        Identify storms in the given DBZ image at the specified timestamp.

        Args:
            dbz_img (np.ndarray): The DBZ image.
            time_frame (datetime): The timestamp of the image.
            map_id (str): The identifier for the storm map. Use for prefixing storm IDs.
            threshold (float): The DBZ threshold for storm identification.
            filter_area (float): The minimum area to filter storms.

        Returns:
            DbzStormsMap: The identified storms map.
        """
        polygons = convert_contours_to_polygons(self.identifier.identify_storm(dbz_map=dbz_map, threshold=threshold, filter_area=filter_area))
        polygons = sorted(polygons, key=lambda x: x.area, reverse=True)
        storms = []

        for idx, polygon in enumerate(polygons):
            contour = convert_polygons_to_contours([polygon])[0]

            # Create the mask of current storm
            mask = np.zeros_like(dbz_map, dtype=np.uint8)
            cv2.fillPoly(mask, contour, color=1)

            # Extract DBZ values inside mask
            weights = dbz_map * mask
            y_idx, x_idx = np.indices(dbz_map.shape)
            total_weight = weights.sum()

            if total_weight == 0:
                centroid = (np.nan, np.nan)  # or fallback
            else:
                cx = (x_idx * weights).sum() / total_weight
                cy = (y_idx * weights).sum() / total_weight
                centroid = (int(cx), int(cy))

            # storms.append(CentroidStorm(polygon, centroid=centroid, id=f"{map_id}_storm_{idx}", img_shape=dbz_map.shape[:2]))
            storms.append(StormObject(polygon, centroid=centroid, id=f"{map_id}_storm_{idx}"))
            
        return StormsMap(storms=storms, time_frame=time_frame, dbz_map=dbz_map)

    def processing_map(self, curr_storms_map: StormsMap) -> int:
        if self.storms_maps == []:
            self.storms_maps.append(curr_storms_map)
            self.tracker = TrackingHistory(curr_storms_map)
            assignments = []
        else:
            prev_storms_map = self.storms_maps[-1]
            dt = (curr_storms_map.time_frame - prev_storms_map.time_frame).seconds / 3600   # scaled to hour

            # match using Hungarian algorithm
            assignments = self.matcher.match_storms(prev_storms_map, curr_storms_map)

            # resolve merge & split
            ## mapping: dict where key -> index of storm; value -> list of tuple[storm_id]
            mapping_prev = {}
            mapping_curr = {}
            for prev_idx, curr_idx in assignments:
                if int(prev_idx) not in mapping_prev:
                    mapping_prev[int(prev_idx)] = []
                mapping_prev[int(prev_idx)].append(int(curr_idx))

                if int(curr_idx) not in mapping_curr:
                    mapping_curr[int(curr_idx)] = []
                mapping_curr[int(curr_idx)].append(int(prev_idx))

            prev_assigned, curr_assigned = mapping_prev.keys(), mapping_curr.keys()
            prev_unassigned = [idx for idx in range(len(prev_storms_map.storms)) if idx not in prev_assigned]
            curr_unassigned = [idx for idx in range(len(curr_storms_map.storms)) if idx not in curr_assigned]


            if len(prev_unassigned) > 0 or len(curr_unassigned) > 0:  # if any unassigned => resolve
                pred_storms_map = StormsMap([
                        self.tracker.forecast(storm.id, dt)
                        for storm in prev_storms_map.storms
                    ], time_frame=curr_storms_map.time_frame)

                # Check for merging
                for prev_idx in prev_unassigned:
                    pred_storm = pred_storms_map.storms[prev_idx]

                    # Find storms that the predicted centroid fall into.
                    candidates = [
                            idx for idx, storm in enumerate(curr_storms_map.storms) \
                                if storm.contour.contains(Point(pred_storm.centroid))
                        ]
                    
                    # Case: more than 1 candidates => choose one with maximum overlapping on prev_storm
                    if len(candidates) > 1:
                        def compute_overlapping(pol: Polygon):
                            return pred_storm.contour.intersection(pol) / pred_storm.contour.area
                        
                        max_idx = np.argmax([compute_overlapping(curr_storms_map.storms[j].contour) \
                                                for j in candidates])
                        candidates = [candidates[max_idx]]
                    
                    mapping_prev[prev_idx] = candidates
                    for cand_idx in candidates:
                        if cand_idx not in mapping_curr:
                            mapping_curr[cand_idx] = []
                        mapping_curr[cand_idx].append(prev_idx)
                
                # Check for splitting
                for curr_idx in curr_unassigned:
                    curr_storm = curr_storms_map.storms[curr_idx]
                    # Find predicted storms that the current centroid fall into.
                    candidates = [
                            idx for idx, storm in enumerate(pred_storms_map.storms) \
                                if storm.contour.contains(Point(curr_storm.centroid))
                        ]
                    
                    # Case: more than 1 candidates => choose one with maximum overlapping on prev_storm
                    if len(candidates) > 1:
                        def compute_overlapping(pol: Polygon):
                            return curr_storm.contour.intersection(pol).area / curr_storm.contour.area
                        
                        max_idx = np.argmax([compute_overlapping(pred_storms_map.storms[i].contour) \
                                                for i in candidates])
                        candidates = [candidates[max_idx]]
                    
                    mapping_curr[curr_idx] = [cand_idx for cand_idx in candidates]
                    for cand_idx in candidates:
                        if cand_idx not in mapping_prev:
                            mapping_prev[cand_idx] = []
                        mapping_prev[cand_idx].append(curr_idx)

            self.tracker.update(mapping_prev, mapping_curr, prev_storms_map, curr_storms_map)

            # Update history movements to track history movement
            for storm in curr_storms_map.storms:
                storm_controller = self.tracker._get_track(storm.id)[0]
                storm.contour_color = storm_controller["storm_lst"][-1].contour_color if len(storm_controller) >= 1 else storm.contour_color
                storm.history_movements = [mv * dt for mv in storm_controller["movement"]]
            self.storms_maps.append(curr_storms_map)
            
        right_matches = list(set([curr for _, curr in assignments]))
        return len(right_matches)