from enum import Enum, auto

class MetricType(Enum):
    MINKOWSKI = 1
    SCHWARZSCHILD = 2
    ELLIS_BRONNIKIVA = 3
    KERR_NEWMAN = 4

class VectorType(Enum):
    COORDINATES = auto()
    IMPULSE = auto()
    IMPULSE_PHOTON_COV = auto()
    IMPULSE_PHOTON_CONTRA = auto()
    DIRECTIONAL_IMPULSE = auto()