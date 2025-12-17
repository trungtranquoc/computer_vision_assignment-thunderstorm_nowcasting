from .base import BaseStormIdentifier
import cv2
import numpy as np

class MorphContourIdentifier(BaseStormIdentifier):
    def __init__(self, n_thresh: int = 3, center_filter: int = 10, kernel_size: int=3):
        self.n_thresh = n_thresh
        self.center_filter = center_filter
        self.kernel_size = kernel_size

    def identify_storm(self, dbz_map: np.ndarray, threshold: int = 30, filter_area: int = 30) -> list[np.ndarray]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size,self.kernel_size))

        # erode for removing weak connection
        lowest_mask = cv2.erode((dbz_map > threshold).astype(np.uint8), kernel)
        num_labels, labels = cv2.connectedComponents(lowest_mask, connectivity=8)

        masks = []
        for label in range(1, num_labels):
            mask = cv2.dilate((labels == label).astype(np.uint8), kernel)
            if np.sum(mask) > filter_area:
                masks.append(mask)

        for i in range(2, self.n_thresh+1):
            current_masks = []
            for roi_mask in masks:
                if np.sum(roi_mask) < self.center_filter:
                    current_masks.append(roi_mask)
                    continue
                current_masks.extend(self._extract_substorms(dbz_map, roi_mask, kernel, threshold=threshold + 5*(i-1), area_filter=filter_area))

            masks = current_masks

        masks = sorted(masks, key=lambda x: np.sum(x), reverse=True)
        return [cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0] for mask in masks]

    def _competitive_dilation(self, sub_masks, outer_mask, kernel, max_iter=200):
        """
        Dilate multiple sub-storm masks simultaneously within outer_mask,
        ensuring no overlap between them.
        """
        current_masks = sub_masks.copy()
        combined_mask = np.clip(np.sum(current_masks, axis=0), 0, 1).astype(np.uint8)

        for _ in range(max_iter):
            grown_masks = []
            changed = False

            for mask in current_masks:
                dilated = cv2.dilate(mask, kernel, iterations=1)
                # only allow dilation into available space (not occupied by others)
                new_mask = dilated & (1 - combined_mask) & outer_mask
                new_mask = np.clip(mask + new_mask, 0, 1)
                grown_masks.append(new_mask)
                if np.any(new_mask != mask):
                    combined_mask = np.clip(np.sum([combined_mask, new_mask], axis=0), 0, 1).astype(np.uint8)
                    changed = True

            current_masks = grown_masks
            # combined_mask = np.clip(np.sum(current_masks, axis=0), 0, 1).astype(np.uint8)

            if not changed:
                break

        return current_masks

    def _extract_substorms(self, dbz_map: np.ndarray, outer_mask: np.ndarray, kernel: np.ndarray, threshold: int, area_filter: int = 10):
        """
        Args:
            dbz_map (np.ndarray): input dbz map.
            outer_mask (np.ndarray): a binary mask indicating the current interest region.
            kernel (np.ndarray): a kernel used for erosion and dilation.
            threshold (int): dbz threshold.
            area_filter (int): if total region under this area => filter out.
        """
        storm_mask = ((dbz_map > threshold) & (outer_mask > 0)).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(storm_mask, connectivity=8)
        sub_masks = []

        for label in range(1, num_labels):
            core_mask = (labels == label).astype(np.uint8)
            if np.sum(core_mask) > area_filter:
                sub_masks.append(core_mask)     # only append if area is large enough

        if len(sub_masks) == 0:
            return [outer_mask]

        return self._competitive_dilation(sub_masks, outer_mask=outer_mask, kernel=kernel)