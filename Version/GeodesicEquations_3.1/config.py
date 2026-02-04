from enums import MetricType

file_mapping = {
    MetricType.MINKOWSKI:               None,
    MetricType.SCHWARZSCHILD:           "CodeOpenCl\\OpenClKernelSchwarzschild.cl",
    MetricType.ELLIS_BRONNIKIVA:        "CodeOpenCl\\OpenClKernelEllis.cl",
    MetricType.KERR_NEWMAN:             "CodeOpenCl\\OpenClKernelKerr.cl",
    MetricType.GOEDEL:                  "CodeOpenCl\\OpenClKernelGoedel.cl",
    MetricType.FRIEDMAN_ROBERTSON:      "CodeOpenCl\\OpenClKernelFriedmanRobertson.cl",
    MetricType.SPHERICAL_UNIVERSE:      "CodeOpenCl\\OpenClKernelSphericalUniverseNEW.cl",
    MetricType.CYLINDRICAL_UNIVERSE:    "CodeOpenCl\\OpenClKernelCylindricalUniverse.cl"
}

# Флаги и настройки
PYOPENCL_CTX = '0'
PYOPENCL_NO_CACHE = '1'

