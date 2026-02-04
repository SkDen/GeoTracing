
import math
import numpy as np
from enums import MetricType
from metric import Metric
from vector4 import Vector4
from tqdm import tqdm

class Camera:
    def __init__(self, width=10, height=10, focus=1.0, aspect_ratio_inv=False, metric=Metric(MetricType.SCHWARZSCHILD)):
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
        self.aspect_ratio_inv = aspect_ratio_inv 
        
        if self.aspect_ratio_inv == False:
            self.aspect_ratio = self.width / self.height
        else:
            self.aspect_ratio = self.height / self.width

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
        
        return np.dot(ratate_phi, ratate_theta)

    def get_pixel_coordinates(self, i, j):
        """
        Преобразует пиксельные координаты (i, j) в нормализованные координаты камеры (u, v)
        """
        if self.aspect_ratio_inv == False:
            u = (2 * i - self.width) / self.width * self.aspect_ratio
            v = (2 * j - self.height) / self.height
        else:
            u = (2 * i - self.width) / self.width
            v = (2 * j - self.height) / self.height * self.aspect_ratio

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

    def create_camera_rays(self, camera_position, energy=1.0, show_progress=True,
                            dtype=np.float64):
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
        global_impulse_contra = np.zeros_like(impulse_array)   # создаем пустой массив котравариантный
        global_impulse_cov = np.zeros_like(impulse_array)

        # Инициализируем прогресс-бар
        rows_range = range(self.height)
        if show_progress:
            rows_range = tqdm(rows_range, desc="Построение нчальных условий камеры: ", unit=" строк")

        # Векторизованное преобразование для всей сетки
        for j in rows_range:
            for i in range(self.width):

                global_impulse_contra[j, i] = self.metric.local_to_global_vector_cont_cont(
                    cam_pos_array,
                    impulse_array[j, i]
                )
                global_impulse_cov[j, i] = self.metric.vector_contra_to_cov(cam_pos_array, global_impulse_contra[j, i])
        
        # Формирование начальных состояний
        initial_states = np.zeros((self.height, self.width, 8), dtype=dtype)
        
        # Заполняем координаты камеры (одинаковые для всех лучей)
        initial_states[..., :4] = cam_pos_array
        
        # Заполняем импульсы
        initial_states[..., 4:] = global_impulse_cov
        
        return initial_states