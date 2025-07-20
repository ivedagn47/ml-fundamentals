from abc import ABC, abstractmethod

class Geometry(ABC):
    """
    Abstract base class for geometry.
    Handles definition of domain geometry for PDE problems.
    """

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, t_min: float, t_max: float):
        """
        Args:
            x_min, x_max: Domain limits in x-direction
            y_min, y_max: Domain limits in y-direction
            t_min, t_max: Time domain limits
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_min = t_min
        self.t_max = t_max

    @abstractmethod
    def domain_bounds(self):
        """Returns the spatial and temporal domain bounds"""
        pass