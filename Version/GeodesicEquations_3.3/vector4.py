
import numpy as np
from numbers import Number
from enums import VectorType

class Vector4:
    """
    Класс для работы с 4-мерными векторами, используемыми в релятивистской физике.
    Поддерживает векторы координат и импульса.
    
    Параметры конструктора:
        t, x, y, z: Компоненты вектора
        vtype: Тип вектора (из перечисления VectorType)
        dtype: Тип данных NumPy (по умолчанию np.float32)
    """
    def __init__(self, t=0, x=0, y=0, z=0, vtype=VectorType.COORDINATES, dtype=np.float32):
        # Проверка типов всех компонент
        if not all(isinstance(c, (int, float)) for c in (t, x, y, z)):
            raise TypeError("Все компоненты вектора должны быть числами")
        
        self.vtype = vtype
        self.dtype = dtype

        # Обработка специальных типов векторов
        if vtype == VectorType.DIRECTIONAL_IMPULSE:
            # Нормировка пространственных компонент
            spatial = np.array([x, y, z], dtype=float)
            norm = np.linalg.norm(spatial)
            
            if norm > 1e-12:  # Порог для избежания деления на ноль
                spatial /= norm
            else:
                raise ValueError("Невозможно нормировать нулевой вектор")
                
            self._data = np.array([t, *spatial], dtype=dtype)

        elif vtype == VectorType.IMPULSE_PHOTON_COV:
            # Нормировка пространственных компонент
            spatial = np.array([x, y, z], dtype=float)
            norm = np.linalg.norm(spatial)
            
            if norm > 1e-12:  # Порог для избежания деления на ноль
                spatial /= norm
            else:
                raise ValueError("Невозможно нормировать нулевой вектор")
                
            self._data = np.array([-1, *spatial], dtype=dtype)

        elif vtype == VectorType.IMPULSE_PHOTON_CONTRA:
            # Нормировка пространственных компонент
            spatial = np.array([x, y, z], dtype=float)
            norm = np.linalg.norm(spatial)
            
            if norm > 1e-12:  # Порог для избежания деления на ноль
                spatial /= norm
            else:
                raise ValueError("Невозможно нормировать нулевой вектор")
                
            self._data = np.array([1, *spatial], dtype=dtype)

        else:
            self._data = np.array([t, x, y, z], dtype=dtype)

    def __mul__(self, other):
        """Умножение вектора на скаляр или матрицу преобразования"""
        if isinstance(other, Number):
            # Умножение на скаляр - возвращаем новый вектор
            return Vector4(*(self._data * other), 
                          vtype=self.vtype,
                          dtype=self.dtype)
            
        elif isinstance(other, np.ndarray):
            # Умножение на матрицу преобразования
            if other.shape == (4, 4):
                return Vector4(*(other @ self._data), 
                              vtype=self.vtype,
                              dtype=self.dtype)
            raise ValueError("Матрица должна иметь размер 4x4")
            
        raise TypeError(f"Неподдерживаемый тип операнда: {type(other)}")

    def __rmul__(self, other):
        """Умножение справа (скаляр или матрица)"""
        return self.__mul__(other)
    
    def __getitem__(self, index):
        """Доступ к компонентам по индексу [0-3]"""
        return self._data[index]
    
    def __setitem__(self, index, value):
        """Установка компонент по индексу"""
        self._data[index] = value

    def __str__(self):
        """Строковое представление вектора"""
        return f"Vector4({self[0]}, {self[1]}, {self[2]}, {self[3]}, type={self.vtype.name})"
    
    def to_array(self):
        """
        Получение массива numpy из компонент вектора
        """
        return np.array([self.t, self.x, self.y, self.z], dtype=self.dtype)
    
    def dot(self, other, metric=(-1, -1, -1)):
        """
        Скалярное произведение 4-векторов с учетом метрики Минковского
        
        Параметры:
            other: Другой Vector4
            metric: Кортеж знаков метрики для пространственных компонент
                    По умолчанию: (-1, -1, -1) для сигнатуры (+, -, -, -)
        """
        if not isinstance(other, Vector4):
            raise TypeError("Ожидается объект Vector4")
            
        # Вычисление произведения с метрикой
        result = self[0] * other[0]
        for i in range(1, 4):
            result += metric[i-1] * self[i] * other[i]
            
        return result

    def normalize_spatial(self):
        """Нормирует пространственные компоненты вектора (x, y, z)"""
        spatial_norm = np.linalg.norm(self._data[1:])
        
        if spatial_norm > 1e-12:
            self._data[1:] /= spatial_norm
        else:
            raise ValueError("Невозможно нормировать нулевые пространственные компоненты")
        return self

    @property
    def t(self):
        return self._data[0]
    
    @property
    def x(self):
        return self._data[1]
    
    @property
    def y(self):
        return self._data[2]
    
    @property
    def z(self):
        return self._data[3]
    
    @property
    def spatial(self):
        """Пространственные компоненты (x, y, z)"""
        return self._data[1:].copy()
    
    @classmethod
    def photon_impulse(cls, energy, direction, dtype=np.float32):
        """
        Создает вектор импульса фотона
        
        Параметры:
            energy: Энергия фотона
            direction: Кортеж (x, y, z) направления движения
            dtype: Тип данных (по умолчанию float32)
        """
        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        
        if norm < 1e-12:
            raise ValueError("Направление не может быть нулевым")
            
        direction = direction / norm
        return cls(energy, 
                  energy * direction[0],
                  energy * direction[1],
                  energy * direction[2],
                  vtype=VectorType.IMPULSE,
                  dtype=dtype)