from functools import reduce
import numpy as np
from sklearn.linear_model import LinearRegression

def linear_tracking_error(movement_history: list[tuple[float, float]]) -> float:
    """
    Compute the linear fitting error distance of final point to the fitted line of previous points.
    """
    points = reduce(lambda points, movement: points + [(points[-1][0] + movement[0], points[-1][1] + movement[1])], movement_history, [(0, 0)])

    # Compute the linear fit parameters
    points = np.array(points)
    x = points[:-1, 0]
    y = points[:-1, 1]

    model = LinearRegression().fit(x.reshape(-1, 1), y)

    # Compute the distance from the final point to the fitted line
    final_point = points[-1]

    return abs(final_point[1] - (model.coef_[0] * final_point[0] + model.intercept_)) / np.sqrt(model.coef_[0]**2 + 1)