"""
Пакет для релятивистских вычислений с использованием OpenCL.
"""

# Импорты для удобства пользователей пакета
from .enums import MetricType, VectorType
from .vector4 import Vector4
from .config import file_mapping, PYOPENCL_CTX

# Определение того, что будет доступно при импорте через *
__all__ = [
    'MetricType',
    'VectorType',
    'Vector4',
    'file_mapping',
    'PYOPENCL_CTX'
]

# Версия пакета
__version__ = "0.1.0"

# Инициализация пакета (если требуется)
import os
os.environ.setdefault('PYOPENCL_CTX', PYOPENCL_CTX)