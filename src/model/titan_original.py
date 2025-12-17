import cv2
import numpy as np
from datetime import datetime, timedelta

from src.cores.base import StormsMap, StormObject, TrackingHistory, UpdateType
from src.identification import MorphContourIdentifier
from src.matcher import MatchedStormPair, SimpleMatcher
from src.utils import convert_contours_to_polygons, convert_polygons_to_contours, spectral_colors

class TitanPrecipitationModel:
    """
    Titan model implementation for thunderstorm nowcasting.
    """
    identifier: MorphContourIdentifier
    storms_maps: list[StormsMap]
    tracker: TrackingHistory
    matcher: SimpleMatcher

    def __init__(self, identifier: MorphContourIdentifier, max_velocity: float = 500):
        self.identifier = identifier
        self.storms_maps = []
        self.matcher = SimpleMatcher(max_velocity=max_velocity)
        self.tracker = None

    def _dynamic_max_velocity(self, area: float) -> float:
        """
        Dynamic constraint for maximum velocity based on storm area. The unit of velocity is pixel/hr.
        """
        if area < 300:
            return 100
        elif area < 500:
            return 200
        else:
            return 300

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

        colors = spectral_colors(len(polygons))

        for idx, polygon in enumerate(polygons):
            contour = convert_polygons_to_contours([polygon])[0]
            color = colors[idx]

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
                centroid = (int(cy), int(cx))

            # storms.append(CentroidStorm(polygon, centroid=centroid, id=f"{map_id}_storm_{idx}", img_shape=dbz_map.shape[:2]))
            storms.append(StormObject(polygon, centroid=centroid, id=f"{map_id}_storm_{idx}", contour_color=color))
            
        return StormsMap(storms=storms, time_frame=time_frame, dbz_map=dbz_map)

    def processing_map(self, curr_storms_map: StormsMap) -> int:
        if self.storms_maps == []:
            self.storms_maps.append(curr_storms_map)
            self.tracker = TrackingHistory(curr_storms_map)
            matched = []
        else:
            prev_storms_map = self.storms_maps[-1]
            dt = (curr_storms_map.time_frame - prev_storms_map.time_frame).seconds / 3600   # scaled to hour

            # match using Hungarian algorithm
            matched: list[MatchedStormPair] = self.matcher.match_storms(prev_storms_map, curr_storms_map)

            for info in matched:
                # print(f"Matched: Prev Storm {prev_storms_map.storms[info.prev_storm_order].id} -> Curr Storm {curr_storms_map.storms[info.curr_storm_order].id} | Update Type: {info.update_type.name}")
                if info.update_type == UpdateType.NEW:
                    self.tracker.add_new_track(
                        new_storm=curr_storms_map.storms[info.curr_storm_order],
                        time_frame=curr_storms_map.time_frame
                    )
                else:
                    self.tracker.update_track(
                        prev_storm=prev_storms_map.storms[info.prev_storm_order],
                        curr_storm=curr_storms_map.storms[info.curr_storm_order],
                        update_type=info.update_type,
                        time_frame=curr_storms_map.time_frame,
                        velocity=info.derive_motion_vector(dt=dt)
                    )

            self.storms_maps.append(curr_storms_map)

    def forecast(self, dt: float):
        current_map = self.storms_maps[-1]
        new_storms = []
        for storm in current_map.storms:
            new_storms.append(storm.forecast(dt))
        
        return StormsMap(storms=new_storms, time_frame=current_map.time_frame + timedelta(hours=dt), dbz_map=None)