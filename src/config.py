from enums import MetricType

file_mapping = {
    MetricType.MINKOWSKI:               "CodeOpenCl\\OpenClKernelMinkovsky.cl",
    MetricType.SCHWARZSCHILD:           "CodeOpenCl\\OpenClKernelSchwarzschild.cl",
    MetricType.ELLIS_BRONNIKIVA:        "CodeOpenCl\\OpenClKernelEllis.cl",
    MetricType.KERR_NEWMAN:             "CodeOpenCl\\OpenClKernelKerr.cl",
    MetricType.GOEDEL:                  "CodeOpenCl\\OpenClKernelGoedel.cl",
    MetricType.FRIEDMAN_ROBERTSON:      "CodeOpenCl\\OpenClKernelFriedmanRobertson.cl",
    MetricType.SPHERICAL_UNIVERSE:      "CodeOpenCl\\OpenClKernelSphericalUniverse.cl",
    MetricType.CYLINDRICAL_UNIVERSE:    "CodeOpenCl\\OpenClKernelCylindricalUniverse.cl",
    MetricType.PARAMETERIZED_WORMHOLE:  "" 
}

file_ray_tracing = {
    MetricType.MINKOWSKI:               "OpenCLRayTracingCores\\RayTracingMinkovsky.cl",
    MetricType.SCHWARZSCHILD:           "OpenCLRayTracingCores\\RayTracingSchwarzschild.cl",
    MetricType.ELLIS_BRONNIKIVA:        "OpenCLRayTracingCores\\RayTracingEllis.cl",
    MetricType.KERR_NEWMAN:             "OpenCLRayTracingCores\\RayTracingKerr.cl",
    MetricType.GOEDEL:                  "",
    MetricType.FRIEDMAN_ROBERTSON:      "",
    MetricType.SPHERICAL_UNIVERSE:      "",
    MetricType.CYLINDRICAL_UNIVERSE:    "",
    MetricType.PARAMETERIZED_WORMHOLE:  "OpenCLRayTracingCores\\RayTracingParameterizedWormhole.cl"
}

# Флаги и настройки
PYOPENCL_CTX = '0'
PYOPENCL_NO_CACHE = '1'

