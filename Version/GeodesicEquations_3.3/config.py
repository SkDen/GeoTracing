from enums import MetricType

file_mapping = {
    MetricType.MINKOWSKI:               "CodeOpenCl\\OpenClKernelMinkovsky.cl",
    MetricType.SCHWARZSCHILD:           "CodeOpenCl\\OpenClKernelSchwarzschild.cl",
    MetricType.ELLIS_BRONNIKIVA:        "CodeOpenCl\\OpenClKernelEllis.cl",
    MetricType.KERR_NEWMAN:             "CodeOpenCl\\OpenClKernelKerr.cl",
    MetricType.GOEDEL:                  "CodeOpenCl\\OpenClKernelGoedel.cl",
    MetricType.FRIEDMAN_ROBERTSON:      "CodeOpenCl\\OpenClKernelFriedmanRobertson.cl",
    MetricType.SPHERICAL_UNIVERSE:      "CodeOpenCl\\OpenClKernelSphericalUniverse.cl",
    MetricType.CYLINDRICAL_UNIVERSE:    "CodeOpenCl\\OpenClKernelCylindricalUniverse.cl"
}

file_ray_tracing = {
    MetricType.MINKOWSKI:               None,
    MetricType.SCHWARZSCHILD:           "OpenCLRayTracingCores\\RayTracingSchwarzschild.cl",
    MetricType.ELLIS_BRONNIKIVA:        "",
    MetricType.KERR_NEWMAN:             "",
    MetricType.GOEDEL:                  "",
    MetricType.FRIEDMAN_ROBERTSON:      "",
    MetricType.SPHERICAL_UNIVERSE:      "",
    MetricType.CYLINDRICAL_UNIVERSE:    ""
}

# Флаги и настройки
PYOPENCL_CTX = '0'
PYOPENCL_NO_CACHE = '1'

