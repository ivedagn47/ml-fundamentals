from core.geometry.geometry import Geometry

class Geometry2D(Geometry):
    """
    Concrete class for 2D rectangular geometry for convection-diffusion equations.
    """

    def __init__(self):
        # Fixed domain as in dataset code
        super().__init__(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, t_min=0.0, t_max=1.0)

    def domain_bounds(self):
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "t_min": self.t_min,
            "t_max": self.t_max,
        }

'''# Example usage
if __name__ == "__main__":
    geometry = Geometry2D()'''