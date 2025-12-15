import numpy as np
from shapely.geometry import Polygon
from typing import Union

def convert_contours_to_polygons(contours: list[Union[np.ndarray, list[np.ndarray]]]) -> list[Polygon]:
    """
        Convert the list of contours into the list of polygons.
    """
    if not contours:
        return []
    if isinstance(contours[0], list):
        contours = [contour for subcontours in contours for contour in subcontours]
    
    polygons = []
    for contour in contours:
        polygon = Polygon(contour.squeeze(axis=1))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.geom_type == "MultiPolygon":
            polygons.extend(list(polygon.geoms))
        else:
            polygons.append(polygon)
    
    return polygons

def convert_polygons_to_contours(polygons: list[Polygon]) -> list[np.ndarray]:
    """
        Convert a list shapely Polygon back into a list of numpy contour format (N, 1, 2).
    """
    contours = []
    for polygon in polygons:
        if polygon.is_empty:
            continue

        # Take exterior coords (skip last point since shapely closes it automatically)
        coords = np.array(polygon.exterior.coords[:-1], dtype=np.int32)
        
        # Reshape into OpenCV contour format
        contours.append(coords.reshape(-1, 1, 2))

    return contours
