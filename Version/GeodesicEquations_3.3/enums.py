from enum import Enum, auto

class MetricType(Enum):
    MINKOWSKI = 1
    SCHWARZSCHILD = 2
    ELLIS_BRONNIKIVA = 3
    KERR_NEWMAN = 4
    GOEDEL = 5
    FRIEDMAN_ROBERTSON = 6
    SPHERICAL_UNIVERSE = 7
    CYLINDRICAL_UNIVERSE = 8

class VectorType(Enum):
    COORDINATES = auto()
    IMPULSE = auto()
    IMPULSE_PHOTON_COV = auto()
    IMPULSE_PHOTON_CONTRA = auto()
    DIRECTIONAL_IMPULSE = auto()

class CoordinatesType(Enum):
    CARTESIAN = auto()
    SPHERICAL = auto()
    CYLINDRICAL = auto()
    HYPERSPHERIC = auto()
    SPECIAL_CYLINDRICAL = auto()