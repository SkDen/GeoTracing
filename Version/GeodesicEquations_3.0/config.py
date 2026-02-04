from enums import MetricType

file_mapping = {
    MetricType.MINKOWSKI: None,
    MetricType.SCHWARZSCHILD: "CodeOpenCl\\OpenClKernelSchwarzschild.cl",
    MetricType.ELLIS_BRONNIKIVA: "CodeOpenCl\\OpenClKernelEllis.cl",
    MetricType.KERR_NEWMAN: "CodeOpenCl\\OpenClKernelKerr.cl"
}

# Флаги и настройки
PYOPENCL_CTX = '0'
PYOPENCL_NO_CACHE = '1'

