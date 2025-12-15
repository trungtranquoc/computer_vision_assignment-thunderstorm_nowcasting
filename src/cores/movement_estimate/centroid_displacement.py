from src.cores.base import StormObject

def centroid_displacement(storm1: StormObject, storm2: StormObject) -> float:
    """
    Calculate the centroid displacement between 2 storms.
    """
    y1, x1 = storm1.centroid.coords[0]
    y2, x2 = storm2.centroid.coords[0]

    return ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5