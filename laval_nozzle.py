import typing
import gdf
import numpy as np


class LavalNozzleSolver:
    def __init__(self, area: typing.Callable[[float], float], x1, x2, num=100):
        self.area = area
        self.x1 = x1
        self.x2 = x2
        self.num = num
        self.x_arr = np.linspace(x1, x2, num)
        self.area_arr = np.array([area(x) for x in self.x_arr])

    def _compute_nominal(self):
        pass

    def compute(self):
        pass
