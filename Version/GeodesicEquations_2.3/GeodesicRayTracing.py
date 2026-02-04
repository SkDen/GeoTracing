
# %%memit
# %load_ext memory_profiler

import os
import sys

# Авто определение параметров
# os.environ["PYOPENCL_NO_CACHE"] = "1"
os.environ['PYOPENCL_CTX'] = '0' 

import pyopencl as cl
import numpy as np
import time
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from memory_profiler import profile
from numbers import Number
from enum import Enum, auto


class VectorType(Enum):
    """Типы 4-векторов для пространства Минковского"""
    COORDINATES = auto()            # Вектор координат (t, x, y, z)
    IMPULSE = auto()                # Вектор импульса (E, px, py, pz)
    IMPULSE_PHOTON_COV = auto()     # Ковариантный вектор импульса фотона (-E, px, py, pz) с энергией E=1
    IMPULSE_PHOTON_CONTRA = auto()  # Ковариантный вектор импульса фотона ( E, px, py, pz) с энергией E=1
    DIRECTIONAL_IMPULSE = auto()    # Нормированный вектор импульса

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


class Metric:
    """Класс для работы с различными метриками пространства-времени"""
    def __init__(self, metric_type='schwarzschild', r_s=1.0, r_0=1.0):
        """
        Инициализация метрики
        
        Параметры:
            metric_type: тип метрики
            r_s: радиус Шварцшильда для метрики Шварцшильда
            r_0: ширина горловины кротовой норы Эллиса
        """
        self.metric_type = metric_type
        self.r_s = r_s
        self.r_0 = r_0
        self.transform_cache = {}
    
    def get_metric_tensor(self, position, dtype=np.float64):
        """
        Возвращает метрический тензор в заданной точке
        
        Параметры:
            position: 4-вектор положения (t, r, θ, φ)
            dtype: тип данных тензора
            
        Возвращает:
            g: метрический тензор (4x4)
        """
        if self.metric_type == 'minkowski':
            return self.minkowski(dtype)
        elif self.metric_type == 'schwarzschild':
            return self.schwarzschild(position, self.r_s, dtype)
        elif self.metric_type == 'ellis':
            return self.ellis(position, self.r_0, dtype)
        else:
            raise ValueError(f"Неизвестный тип метрики: {self.metric_type}")
    

# ===== Определяем статические методы для получения метрических тензоров =======================
# ==============================================================================================
    @staticmethod
    def minkowski(dtype=np.float32):
        '''Метрический тензор Минковского'''
        return np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=dtype)
    
    @staticmethod
    def schwarzschild(vect_pos, r_s, dtype=np.float32):
        '''Метрический тензор Шварцшильда'''
        r = vect_pos[1]
        theta = vect_pos[2]
        
        # Диагональные компоненты
        gamma = 1 - r_s / r
        g_tt = -gamma
        g_rr = 1 / gamma
        g_theta_theta = r**2
        g_phi_phi = (r * np.sin(theta))**2

        return np.array([
            [g_tt, 0, 0, 0],
            [0, g_rr, 0, 0],
            [0, 0, g_theta_theta, 0],
            [0, 0, 0, g_phi_phi]
        ], dtype=dtype)
    
    @staticmethod
    def ellis(vector_pos, r_0, dtype=np.float32):
        l = vector_pos[1]
        theta = vector_pos[2]

        r_min = l**2 + r_0**2
        g_tt = -1
        g_ll = 1
        g_theta_theta = r_min
        g_phi_phi = r_min * np.sin(theta)**2

        return np.array([
            [g_tt, 0, 0, 0],
            [0, g_ll, 0, 0],
            [0, 0, g_theta_theta, 0],
            [0, 0, 0, g_phi_phi]
        ], dtype=dtype)
# ==============================================================================================
# ==============================================================================================
    
    def _Vector4_array(self, vector):
        # Поддерживаем как Vector4, так и массивы NumPy
        if isinstance(vector, Vector4):
            return vector.to_array()
        else:
            return vector

# ===== Функции для определения скалярного произведения ========================================
# ==============================================================================================
    def scalar_product_contra_contra(self, position, vec1_contra, vec2_contra, dtype=np.float32):
        """
        Скалярное произведение двух контравариантных векторов:
        g_{ij} A^i B^j
        
        Параметры:
            position: точка пространства-времени (4-вектор)
            vec1_contra: первый контравариантный вектор
            vec2_contra: второй контравариантный вектор
            dtype: тип данных вычислений
            
        Возвращает:
            scalar: скалярное произведение
        """

        g = self.get_metric_tensor(position, dtype)
        return self._compute_product(g, vec1_contra, vec2_contra)
    
    def scalar_product_cov_cov(self, position, vec1_cov, vec2_cov, dtype=np.float64):
        """
        Скалярное произведение двух ковариантных векторов:
        g^{ij} A_i B_j
        
        Параметры:
            position: точка пространства-времени (4-вектор)
            vec1_cov: первый ковариантный вектор
            vec2_cov: второй ковариантный вектор
            dtype: тип данных вычислений
            
        Возвращает:
            scalar: скалярное произведение
        """
        g = self.get_metric_tensor(position, dtype)
        g_inv = np.linalg.inv(g)
        return self._compute_product(g_inv, vec1_cov, vec2_cov)
    
    def scalar_product_mixed(self, position, vec_contra, vec_cov, dtype=np.float64):
        """
        Скалярное произведение контравариантного и ковариантного векторов:
        A_i B^j
        
        Параметры:
            position: точка пространства-времени (4-вектор)
            vec_contra: контравариантный вектор
            vec_cov: ковариантный вектор
            dtype: тип данных вычислений
            
        Возвращает:
            scalar: скалярное произведение
        """
        # Преобразуем векторы в массивы
        v1 = self._to_array(vec_contra)
        v2 = self._to_array(vec_cov)
        
        # Для смешанных векторов метрика не требуется
        if v1.ndim == 1 and v2.ndim == 1:
            return np.dot(v1, v2)
        elif v1.ndim == 2 and v2.ndim == 2:
            return np.einsum('...i,...i', v1, v2)
        elif v1.ndim == 2 and v2.ndim == 1:
            return np.dot(v1, v2)
        else:
            raise ValueError("Несовместимые размерности векторов")

    def _compute_product(self, metric_tensor, vec1, vec2):
        """Внутренний метод для вычисления произведения с метрикой"""
        v1 = self._to_array(vec1)
        v2 = self._to_array(vec2)
        
        if v1.ndim == 1 and v2.ndim == 1:
            return np.einsum('i,ij,j', v1, metric_tensor, v2)
        elif v1.ndim == 2 and v2.ndim == 2:
            return np.einsum('...i,ij,...j', v1, metric_tensor, v2)
        elif v1.ndim == 2 and v2.ndim == 1:
            # Векторизованное вычисление для сетки векторов
            return np.einsum('...i,ij,j', v1, metric_tensor, v2)
        else:
            raise ValueError("Несовместимые размерности векторов")
    
    def _to_array(self, vector):
        """Преобразует Vector4 или список в массив NumPy"""
        if isinstance(vector, Vector4):
            return vector.to_array()
        elif isinstance(vector, np.ndarray):
            return vector
        else:
            return np.asarray(vector)
# ==============================================================================================
# ==============================================================================================
    
    def local_to_global_4momentum_cov(self, vector_position_cont, vect_napr_cont):
        """
        Преобразует 4-импульс из локальной системы отсчёта в глобальную
        
        Параметры:
            vector_position: 4-вектор положения (t, r, θ, φ)
            E: энергия частицы
            vect_napr_cont: контравариантные компоненты 4-импульса в локальной системе
        
        Возвращает:
            cov_impulse: ковариантные компоненты 4-импульса в глобальной системе
        """
        # Поддерживаем как Vector4, так и массивы NumPy
        if isinstance(vector_position_cont, Vector4):
            vector_position_cont = vector_position_cont.to_array()
        else:
            vector_position_cont = vector_position_cont
        
        if self.metric_type == 'minkowski':
            # Для метрики Минковского преобразование не требуется
            return vect_napr_cont
        
        elif self.metric_type == 'schwarzschild':
            # Вычисляем метрические коэффициенты
            r = vector_position_cont[1]
            gamma = 1 - self.r_s / r
            sqrt_gamma = np.sqrt(gamma)
            
            # Копируем для безопасности
            vect_napr_cont = np.array(vect_napr_cont, copy=True)
            vect_napr_cov = np.zeros_like(vect_napr_cont)
            
            # Вычисляем контравариантные компоненты 4-импульса
            if vect_napr_cont.ndim == 1:
                vect_napr_cont[0] *= 1 / sqrt_gamma
                vect_napr_cont[1] *= sqrt_gamma
                vect_napr_cont[2] *= 1 / r
                vect_napr_cont[3] *= 1 / (r * np.sin(vector_position_cont[2]))
            else:
                vect_napr_cont[..., 0] *= 1 / sqrt_gamma
                vect_napr_cont[..., 1] *= sqrt_gamma
                vect_napr_cont[..., 2] *= 1 / r
                vect_napr_cont[..., 3] *= 1 / (r * np.sin(vector_position_cont[2]))
            
            # Преобразуем в ковариантные компоненты
            g = self.schwarzschild(vector_position_cont, self.r_s, vect_napr_cont.dtype)
            
            if vect_napr_cont.ndim == 1:
                for i in range(4):
                    vect_napr_cov[i] = np.sum(g[i] * vect_napr_cont)
            else:
                # Векторизованное преобразование
                vect_napr_cov = np.einsum('ij,...j->...i', g, vect_napr_cont)
            
            return vect_napr_cov
        elif self.metric_type == 'ellis':
            # Вычисляем метрические коэффициенты
            sin_theta = np.sin(vector_position_cont[2])
            sqr_l2r_02 = np.sqrt(vector_position_cont[1]**2 + self.r_0**2)

            # Копируем для безопасности
            vect_napr_cont = np.array(vect_napr_cont, copy=True)
            vect_napr_cov = np.zeros_like(vect_napr_cont)

            # Вычисляем контравариантные компоненты 4-импульса
            if vect_napr_cont.ndim == 1:
                vect_napr_cont[0] *= 1
                vect_napr_cont[1] *= 1
                vect_napr_cont[2] *= 1 / sqr_l2r_02
                vect_napr_cont[3] *= 1 / (sqr_l2r_02 * sin_theta)
            else:
                vect_napr_cont[..., 0] *= 1
                vect_napr_cont[..., 1] *= 1
                vect_napr_cont[..., 2] *= 1 / sqr_l2r_02
                vect_napr_cont[..., 3] *= 1 / (sqr_l2r_02 * sin_theta)

            # Преобразуем в ковариантные компоненты
            g = self.ellis(vector_position_cont, self.r_0, vect_napr_cont.dtype)
            
            if vect_napr_cont.ndim == 1:
                for i in range(4):
                    vect_napr_cov[i] = np.sum(g[i] * vect_napr_cont)
            else:
                # Векторизованное преобразование
                vect_napr_cov = np.einsum('ij,...j->...i', g, vect_napr_cont)
            
            return vect_napr_cov
        else:
            raise ValueError(f"Метрика {self.metric_type} не поддерживается для преобразования импульса")
    
    def coordinate_transformation(self, from_system, to_system, vector):
        """
        Преобразует вектор между системами координат
        
        Параметры:
            from_system: исходная система координат ('cartesian', 'spherical')
            to_system: целевая система координат ('cartesian', 'spherical')
            vector: 4-вектор для преобразования
            
        Возвращает:
            transformed_vector: преобразованный 4-вектор
        """
        if from_system == to_system:
            return vector
        
        key = (from_system, to_system)
        if key not in self.transform_cache:
            # Создаем матрицу преобразования
            if key == ('cartesian', 'spherical'):
                # Декартовы -> сферические
                transform_matrix = self.cartesian_to_spherical_matrix(vector)
            elif key == ('spherical', 'cartesian'):
                # Сферические -> декартовы
                transform_matrix = self.spherical_to_cartesian_matrix(vector)
            else:
                raise ValueError(f"Неизвестное преобразование: {from_system} -> {to_system}")
            
            self.transform_cache[key] = transform_matrix
        
        return np.dot(self.transform_cache[key], vector)

    
    @staticmethod
    def cartesian_to_spherical_matrix(vector):
        """Матрица преобразования из декартовых в сферические координаты"""
        x, y, z = vector[1], vector[2], vector[3]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) if r > 0 else 0
        phi = np.arctan2(y, x)
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # Матрица преобразования для пространственных компонент
        spatial_transform = np.array([
            [x/r, y/r, z/r],
            [(x*z)/(r**2*np.sqrt(x**2+y**2)), (y*z)/(r**2*np.sqrt(x**2+y**2)), -np.sqrt(x**2+y**2)/r**2],
            [-y/(x**2+y**2), x/(x**2+y**2), 0]
        ])
        
        # Полная матрица 4x4 (временная компонента не изменяется)
        full_matrix = np.eye(4)
        full_matrix[1:, 1:] = spatial_transform
        
        return full_matrix
    
    @staticmethod
    def spherical_to_cartesian_matrix(vector):
        """Матрица преобразования из сферических в декартовы координаты"""
        r, theta, phi = vector[1], vector[2], vector[3]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # Матрица преобразования для пространственных компонент
        spatial_transform = np.array([
            [sin_theta*cos_phi, r*cos_theta*cos_phi, -r*sin_theta*sin_phi],
            [sin_theta*sin_phi, r*cos_theta*sin_phi, r*sin_theta*cos_phi],
            [cos_theta, -r*sin_theta, 0]
        ])
        
        # Полная матрица 4x4 (временная компонента не изменяется)
        full_matrix = np.eye(4)
        full_matrix[1:, 1:] = spatial_transform
        
        return full_matrix
    

class Camera:
    def __init__(self, width=10, height=10, focus=1.0, metric=Metric('schwarzschild')):
        try:
            self.width = int(width)
            self.height = int(height)
        except (TypeError, ValueError):
            print("Размеры экрана должны быть конвертируемы в целые числа.")
            print(f"Используемые параметры: width={int(width)}, height={int(height)}")
            self.width = int(width)
            self.height = int(height)
        
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Размеры экрана должны быть положительными числами")
        
        self.focus = float(focus)
        self.aspect_ratio = self.width / self.height
        self.metric = metric
        self.theta = 0.0  # Полярный угол (по умолчанию)
        self.phi = 0.0    # Азимутальный угол (по умолчанию)

    def set_direction(self, theta, phi):
        """
        Устанавливает направление камеры с помощью углов
        
        Параметры:
            theta: полярный угол в радианах (-π/2 ≤ theta ≤ π/2)
            phi: азимутальный угол в радианах (0 ≤ phi < 2π)
        """
        if not (-np.pi/2 <= theta <= np.pi/2):
            raise ValueError("theta должен быть в диапазоне [-π/2, π/2]")
        if not (0 <= phi < 2*np.pi):
            raise ValueError("phi должен быть в диапазоне [0, 2π)")
        
        self.theta = theta
        self.phi = phi
    
    def rotation_matrix_camera(self, nu, mu):
        """
        Расчитываем матрицу поворота камеры
         локальная координатная система камеры описывается 
         парметрами ((r), (theta), (phi)) которая эквивалентна декартовым 
         (оси O(r), O(theta), O(phi) ортогональны друг-другу).

        Вращение камеры определяется двумя углами (nu) и (mu):
            nu - определяет вращение вокруг оси O(phi);
            mu - определяет вращение вокруг оси O(theta).
        """
        
        c_phi = np.cos(nu)
        s_phi = np.sin(nu)
        ratate_phi = np.array([[ c_phi, s_phi, 0],
                               [-s_phi, c_phi, 0],
                               [     0,     0, 1]])
        
        c_theta = np.cos(mu)
        s_theta = np.sin(mu)
        ratate_theta = np.array([[ c_theta, 0, s_theta],
                                 [       0, 1,       0],
                                 [-s_theta, 0, c_theta]])
        
        return np.dot(ratate_theta, ratate_phi)

    def get_pixel_coordinates(self, i, j):
        """
        Преобразует пиксельные координаты (i, j) в нормализованные координаты камеры (u, v)
        """
        u = (2 * i - self.width) / self.width * self.aspect_ratio
        v = (2 * j - self.height) / self.height
        return u, v
    
    def get_pixel_indices(self, u, v):
        # Обратное преобразование для i и j (вещественные значения)
        i_val = (u / self.aspect_ratio + 1) * (self.width / 2)
        j_val = (v + 1) * (self.height / 2)
        
        # Функция округления по вашему правилу
        def custom_round(x):
            n = math.floor(x)
            f = x - n
            return n + 1 if f >= 0.5 else n

        # Округляем до ближайшего целого
        i = custom_round(i_val)
        j = custom_round(j_val)
        
        # Обеспечиваем попадание в границы массива
        i = max(0, min(self.width - 1, i))
        j = max(0, min(self.height - 1, j))
        
        return int(i), int(j)

    def create_camera_rays(self, camera_position, energy=1.0, dtype=np.float64):
        """
        Создает массив начальных состояний лучей для камеры
        
        Параметры:
            camera_position: Vector4 или массив - положение камеры
            energy: энергия фотонов (по умолчанию 1.0)
            dtype: тип данных массива
            
        Возвращает:
            initial_states: массив формы (height, width, 8)
        """
        # Преобразуем camera_position в массив
        if isinstance(camera_position, Vector4):
            cam_pos_array = camera_position.to_array()
        elif isinstance(camera_position, np.ndarray):
            cam_pos_array = camera_position
        else:
            raise TypeError("camera_position должен быть Vector4 или numpy массивом")
        
        # Создаем матрицу поворота 3x3
        rotation_matrix = self.rotation_matrix_camera(self.theta, self.phi)
        
        # Создаем сетку координат
        i_indices = np.arange(self.width)
        j_indices = np.arange(self.height)
        i_grid, j_grid = np.meshgrid(i_indices, j_indices)
        u_grid, v_grid = self.get_pixel_coordinates(i_grid, j_grid)
        
        # Вычисляем направления лучей в системе камеры
        directions_cam = np.zeros((self.height, self.width, 3), dtype=dtype)
        directions_cam[..., 0] = -self.focus    # X-компонента (вправо)
        directions_cam[..., 1] = v_grid         # Y-компонента (вверх)
        directions_cam[..., 2] = u_grid         # Z-компонента (вперед)

        # Производим вращение векторов направления
        directions_cam = np.dot(directions_cam, rotation_matrix)

        # Нормализуем направления
        norms = np.linalg.norm(directions_cam, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-12  # Защита от деления на ноль
        directions_cam = directions_cam / norms
        
        # Создаем массив импульсов
        impulse_array = np.zeros((self.height, self.width, 4), dtype=dtype)
        impulse_array[..., 0] = 1                 # Временная компонента
        impulse_array[..., 1:] = directions_cam   # Пространственные компоненты

        # Присваиваем энергию
        impulse_array[..., 0:] *= energy
        
        # Преобразование импульсов в глобальную систему координат с учетом метрики
        global_impulse = np.zeros_like(impulse_array)   # создаем пустой массив
        
        # Векторизованное преобразование для всей сетки
        for j in range(self.height):
            for i in range(self.width):
                global_impulse[j, i] = self.metric.local_to_global_4momentum_cov(
                    cam_pos_array,
                    impulse_array[j, i]
                )
        
        # Формирование начальных состояний
        initial_states = np.zeros((self.height, self.width, 8), dtype=dtype)
        
        # Заполняем координаты камеры (одинаковые для всех лучей)
        initial_states[..., :4] = cam_pos_array
        
        # Заполняем импульсы
        initial_states[..., 4:] = global_impulse
        
        return initial_states


def plot_trajectories(trajectories, point_counts, r_s, W, H, step_ray=1, avto_mashtab=True, black_hole=True):
    """
    Строит 3D график траекторий геодезических
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Преобразование сферических координат в декартовы
    def to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    # Счетчик построенных траекторий
    trajectory_count = 0
    
    # Цикл по всем траекториям с заданным шагом
    for idx in range(0, W*H, step_ray):
        num_points = point_counts[idx]
        if num_points < 2:
            continue
            
        # Извлекаем координаты
        traj = trajectories[idx][:num_points]
        r = traj[:, 1]  # Радиальная координата
        theta = traj[:, 2]  # Угол theta (полярный угол)
        phi = traj[:, 3]  # Угол phi (азимутальный угол)
        
        # Фильтрация точек за горизонтом
        valid = r > 1.001 * r_s
        r = r[valid]
        theta = theta[valid]
        phi = phi[valid]
        
        if len(r) < 2:
            continue
            
        # Преобразование координат
        x, y, z = to_cartesian(r, theta, phi)
        
        # Вычисление цвета на основе индекса траектории
        color_value = idx / (W * H)
        color = plt.cm.viridis(color_value)
        
        # Построение траектории
        ax.plot(x, y, z, 
                linewidth=0.7, 
                alpha=0.8,
                color=color)
        
        trajectory_count += 1

    print(f"Построено траекторий: {trajectory_count}")

    if black_hole == True:
        # Визуализация черной дыры
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_bh = r_s * np.outer(np.cos(u), np.sin(v))
        y_bh = r_s * np.outer(np.sin(u), np.sin(v))
        z_bh = r_s * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_bh, y_bh, z_bh, color='black', alpha=0.7, label='фактический горизонт событий')

        # Фотонная сфера (круг в экваториальной плоскости)
        phi = np.linspace(0, 2*np.pi, 100)
        r_photon = 1.5 * r_s
        x_ph = r_photon * np.cos(phi)
        y_ph = r_photon * np.sin(phi)
        z_ph = np.zeros_like(phi)
        ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7, label='Фотонная сфера')
        
        # Фотонная сфера (круг вокруг оси y)
        x_ph = r_photon * np.cos(phi)
        z_ph = r_photon * np.sin(phi)
        y_ph = np.zeros_like(phi)
        ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7)

        # Фотонная сфера (круг вокруг оси x)
        z_ph = r_photon * np.cos(phi)
        y_ph = r_photon * np.sin(phi)
        x_ph = np.zeros_like(phi)
        ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7)

    # Настройки визуализации
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Траектории света в метрике Шварцшильда\n(3D декартовы координаты)', fontsize=14)
    ax.legend()
    
    # Автомасштабирование
    all_points = trajectories[:, :, 1:4].reshape(-1, 3)
    valid_points = all_points[all_points[:, 0] > 1.001 * r_s]
    
    if avto_mashtab == True:
        if valid_points.size > 0:
            max_val = np.max(valid_points)
            min_val = np.min(valid_points)
            max_range = max(np.abs(max_val), np.abs(min_val)) * 1.1
            
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
    else:
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_zlim(-18, 18)

    
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def OpenCL_platform():
    # Получаем список всех платформ
    platforms = cl.get_platforms()

    print("\nДоступные платформы и устройства:")
    for i, platform in enumerate(platforms):
        print(f"Платформа {i}: {platform.name}")
        print(f"  Вендор: {platform.vendor}")
        print(f"  Версия: {platform.version}")
        
        # Получаем устройства на этой платформе
        devices = platform.get_devices()
        for j, device in enumerate(devices):
            print(f"  Устройство {j}:")
            print(f"    Имя: {device.name}")
            print(f"    Тип: {cl.device_type.to_string(device.type)}")
            print(f"    Макс. размер рабочей группы: {device.max_work_group_size}")
            print(f"    Макс. вычислительные единицы: {device.max_compute_units}")
            print(f"    Глобальная память: {device.global_mem_size/(1024**3):.2f} GB")
            print(f"    Локальная память: {device.local_mem_size/1024:.2f} KB")
            print(f"    Макс. размер буфера: {device.max_mem_alloc_size/(1024**3):.2f} GB")

def OpenCL_using_device(ctx):
    # Выводим информацию о используемых устройств
    print("Устройства в текущем контексте:")
    for i, device in enumerate(ctx.devices):
        print(f"  Устройство {i}: {device.name}")
        print(f"    Тип: {cl.device_type.to_string(device.type)}")
        
        # Получаем информацию о вычислительных возможностях
        if hasattr(device, 'max_work_item_sizes'):
            print(f"    Макс. размер рабочего элемента: {device.max_work_item_sizes}")
        # # Проверяем расширения
        # if hasattr(device, 'extensions'):
        #     print(f"    Поддерживаемые расширения: {device.extensions}")


# @profile
def OpenCL(kernel, initial_states, lambda_0, lambda_end, h, r_s, global_size, max_points=1000, save_step=10, dtype=np.float64):
    
    # Определяем число итераций в ядре для заданного времени моделирования
    iterations = int((lambda_end - lambda_0) / h)
    
    # Определяем максимальное число лучей
    width, height = global_size
    total_rays = width * height

    # Преобразуем двухмерный массив импульсов (i, j) в одномерный
    initial_states_flat = initial_states.reshape(-1, 8)

    # Инициализация контекста OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    # OpenCL_using_device(ctx)

    # Выводим информацию о максимальной памяти
    device = ctx.devices[0]
    max_alloc_size = device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    global_mem_size = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
    print(f"\nМакс. размер буфера: {max_alloc_size/(1024**2):.2f} MB")
    print(f"Общая память: {global_mem_size/(1024**2):.2f} MB")

    # Загрузка кода ядра и компиляция программы
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(script_dir, kernel)
    with open(kernel_path, encoding='utf-8') as f:
        kernel_code = f.read()
    prg = cl.Program(ctx, kernel_code).build()

    # Выделение памяти для результатов
    trajectories = np.zeros((total_rays, max_points, 8), dtype=dtype)
    point_counts = np.zeros(total_rays, dtype=np.int32)

    # Размер батча
    batch_size = min(total_rays, 5000)  # Безопасный размер батча
    batches = (total_rays + batch_size - 1) // batch_size

    # Вычисляем максимальные размеры буферов
    max_initial_size = batch_size * 8 * np.dtype(dtype).itemsize
    max_trajectories_size = batch_size * max_points * 8 * np.dtype(dtype).itemsize
    max_points_size = batch_size * np.dtype(np.int32).itemsize

    # Проверяем ограничения памяти устройства
    if max_trajectories_size > max_alloc_size:
        # Автоматическая коррекция размера батча
        max_batch_size = max_alloc_size // (max_points * 8 * np.dtype(dtype).itemsize)
        max_initial_size = max_batch_size * 8 * np.dtype(dtype).itemsize
        max_trajectories_size = max_batch_size * max_points * 8 * np.dtype(dtype).itemsize
        max_points_size = max_batch_size * np.dtype(np.int32).itemsize
        print(f"Размер батча уменьшен до {max_batch_size} из-за ограничений памяти устройства")

    # Создаем буферы максимального размера
    mf = cl.mem_flags
    initial_buf = cl.Buffer(ctx, mf.READ_WRITE, max_initial_size)
    trajectories_buf = cl.Buffer(ctx, mf.WRITE_ONLY, max_trajectories_size)
    point_counts_buf = cl.Buffer(ctx, mf.WRITE_ONLY, max_points_size)

    # Выводим информацию о размерах буферов
    print(f"\nРазмер буфера начальных условий: {max_initial_size/(1024**2):.2f} МБ")
    print(f"Размер буфера траекторий: {max_trajectories_size/(1024**2):.2f} МБ")
    print(f"Размер буфера счетчиков: {max_points_size/(1024**2):.2f} МБ")

    # Выводим некоторую информацию
    print(f"\nКоличество моделируемых лучей: {total_rays}")
    print(f"Общее число итераций в ядре: {iterations}")
    print(f"Временной интервал моделирования: {h*iterations} сек")
    print(f"Максимальное количество батчей: {batches}")

    # Расчитываем размер выходного массива с траекториями
    total_mb = sys.getsizeof(trajectories) / (1024 * 1024)
    print(f"Размер выходного массива траекторий: {total_mb:.2f} МБ\n")

    start_total = time.time()
    for batch_idx in range(batches):
        start_batch = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_rays)
        current_batch_size = end_idx - start_idx

        # Копируем начальные условия для текущего батча
        initial_batch = initial_states_flat[start_idx:end_idx]
        # Правильное копирование данных на устройство
        cl.enqueue_copy(
            queue, 
            initial_buf, 
            initial_batch
        )

        # Запуск ядра для текущего батча
        prg.runge_kutta4_trajectories(
            queue, 
            (current_batch_size,),  # 1D NDRange
            None,
            dtype(r_s),
            dtype(h),
            np.int32(iterations),
            np.int32(max_points),
            np.int32(save_step),
            initial_buf,
            trajectories_buf,
            point_counts_buf
        )
        
        # Чтение результатов батча
        trajectories_batch_flat = np.empty(current_batch_size * max_points * 8, dtype=dtype)
        point_counts_batch = np.empty(current_batch_size, dtype=np.int32)
        
        # Копирование данных с устройства
        cl.enqueue_copy(
            queue, 
            trajectories_batch_flat, 
            trajectories_buf
        )
        cl.enqueue_copy(
            queue, 
            point_counts_batch, 
            point_counts_buf
        )
        
        # Ожидаем завершения операций
        queue.finish()
        
        # Проверяем размеры перед reshape
        expected_size = current_batch_size * max_points * 8
        if trajectories_batch_flat.size != expected_size:
            raise ValueError(
                f"Несоответствие размеров: ожидалось {expected_size}, "
                f"получено {trajectories_batch_flat.size}"
            )
        
        # Сохранение результатов
        trajectories_batch = trajectories_batch_flat.reshape(current_batch_size, max_points, 8)
        trajectories[start_idx:end_idx] = trajectories_batch
        point_counts[start_idx:end_idx] = point_counts_batch
        
        print(f"Батч {batch_idx+1}/{batches} ({current_batch_size} лучей) занял: {time.time() - start_batch:.2f} сек")
    print(f"Всего вычислено {total_rays} траекторий за {time.time() - start_total:.2f} секунд\n")
    
    # Освобождаем ресурсы
    initial_buf.release()
    trajectories_buf.release()
    point_counts_buf.release()

    return trajectories, point_counts


if __name__ == "__main__":
    # Вывод информации о устройствах пралельного вычисления
    # OpenCL_platform()

    # Коды для вычислительных ядер
    Ellis = 'OpenClKernelEllis.cl'
    Schwarzschild = 'OpenClKernelSchwarzschild.cl'

    # Параметры интегрирования
    r_s = 3.0           # Радиус Шварцшильда
    r_0 = 3.0           # Горловина кротовой норы
    h = 0.01            # Шаг интегрирования
    lambda_0 = 0        # Начало интегрирования
    lambda_end = 50     # Конец интегрирования

    # Парметры камеры
    W = 20
    H = 100    

    # Инициализируем метрики
    metric_ellis = Metric('ellis', r_0=r_0)
    metric_schwarzschild = Metric('schwarzschild', r_s=r_s)

    # Инициализируем камеру (задаем параметры размера, фокуса, тим используемой метрики)
    camera = Camera(width=W, height=H, focus=2, metric=metric_ellis)
    # 4-вектор позиции камеры в сферический координатах
    camera_pos = Vector4(t=0, x=30, y=np.pi/2, z=0, vtype=VectorType.COORDINATES, dtype=np.float64)
    # Задаем поворот камеры
    camera.set_direction(theta=0, phi=0)
    # Инициализируем массив начальных импульсов для фотонов камеры
    initial_states_cov = camera.create_camera_rays(camera_position=camera_pos, energy=1, dtype=np.float64)

    print(metric_ellis.scalar_product_cov_cov(camera_pos, initial_states_cov[..., 0, 4:], initial_states_cov[..., 0, 4:]))

    # Вычисляем траектории
    trajectories, point_counts = OpenCL(
        Ellis,
        initial_states_cov,
        lambda_0,
        lambda_end,
        h,
        r_s,
        (W, H),
        max_points=int((lambda_end - lambda_0) / h),
        save_step=1,
        dtype = np.float64
    )

    np.savetxt('full_array.txt', trajectories[0], delimiter='\t', encoding='utf-8', fmt='%.3f')

    # Визуализация
    plot_trajectories(trajectories, point_counts, r_0, W, H, step_ray=2,
                      avto_mashtab=False, black_hole=True)

# %%
