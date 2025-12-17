import numpy as np
import cv2
from typing import Union

from src.cores.base import StormsMap, StormObject
from src.utils import convert_polygons_to_contours
from .overlapping_score import pod_score, far_score, csi_score

class PredictionBenchmarkModel:
    def __init__(self):
        self.pods = []
        self.fars = []
        self.csis = []
        self.predicted = []

    def compute_success(self, predicted_active: np.ndarray, actual_active: np.ndarray) -> int:
        return np.sum(predicted_active & actual_active)
    
    def compute_failure(self, predicted_active: np.ndarray, actual_active: np.ndarray) -> int:
        return np.sum((~predicted_active) & actual_active)

    def compute_false_alarm(self, predicted_active: np.ndarray, actual_active: np.ndarray) -> int:
        return np.sum(predicted_active & (~actual_active))

    def evaluate_predict(self, predicted_storms_map: Union[StormsMap, np.ndarray], ground_truth: np.ndarray):
        """
        Receive the actual active map and predicted storms map to compute evaluation metrics, using 3 metric scores: POD, FAR, CSI.
        """
        if isinstance(predicted_storms_map, np.ndarray):
            predicted_active = predicted_storms_map.astype(bool)
        else:
            predicted_active = np.zeros(ground_truth.shape, dtype=np.int8)
            # Fill out 1 to pixel that belong to any storm
            for storm in predicted_storms_map.storms:
                cv2.drawContours(predicted_active, convert_polygons_to_contours([storm.contour]), -1, color=1, thickness=-1)
            predicted_active = predicted_active.astype(bool)
        
        success = self.compute_success(predicted_active, ground_truth)
        failure = self.compute_failure(predicted_active, ground_truth)
        false_alarm = self.compute_false_alarm(predicted_active, ground_truth)

        self.pods.append(success / (success + failure) if (success + failure) > 0 else 0)
        self.fars.append(false_alarm / (success + false_alarm) if (success + false_alarm) > 0 else 0)
        self.csis.append(success / (success + failure + false_alarm) if (success + failure + false_alarm) > 0 else 0)