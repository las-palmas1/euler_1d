from enum import Enum


class SpaceScheme(Enum):
    Godunov = 0
    VanLeer = 1
    StegerWarming = 2
    HLLC = 3
    Roe = 4
    RoeEntropyFix = 5


class TimeScheme(Enum):
    ExplicitEuler = 0


class TimeStepping(Enum):
    Global = 0
    Local = 1
