from abc import ABC, abstractmethod
import numpy as np

class BaseStormIdentifier(ABC):
    @abstractmethod
    def identify_storm(self, image: np.ndarray, threshold: float, filter_area: float) -> list[np.ndarray]:
        """
            Identify storm objects from the image.

            Args:
                image (np.ndarray): The input image from which to identify storm objects.
                **args: Additional arguments that may be required for identification.
            Returns:
                List[np.ndarray]: A list of identified storm objects.
        """
        pass