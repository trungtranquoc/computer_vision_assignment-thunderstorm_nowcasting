from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .storm_object import StormObject

THRESHOLD_DBZ = 30

@dataclass
class StormsMap:
    storms: list[StormObject]
    time_frame: datetime
    dbz_map: np.ndarray = None