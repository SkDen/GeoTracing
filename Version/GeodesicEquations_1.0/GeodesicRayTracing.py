
# %%memit
# %load_ext memory_profiler

import os
# os.environ["PYOPENCL_NO_CACHE"] = "1"
# os.environ['PYOPENCL_CTX'] = '0'

import pyopencl as cl
import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from memory_profiler import profile
from numbers import Number
from enum import Enum, auto


class Metric:
    def Minkovsky(bytes=np.float32):
        '''Метрический тензор минковского'''
        Tensor = np.array([[-1, 0, 0, 0],
                           [ 0, 1, 0, 0],
                           [ 0, 0, 1, 0],
                           [ 0, 0, 0, 1]], bytes)
        return Tensor
    
    def Schwarzschild(vect_pos, r_s, bytes=np.float32):
        '''Метрический тензор Шварцшильда.\n
            определяется в точке пространства и времени (t, r, θ, φ)'''
        def Gamma(r, r_s):
            return 1 - r_s/r
        r = vect_pos[1]
        theta = vect_pos[2]
        
        # Диагональные компоненты
        comp11 = Gamma(r, r_s)
        comp22 = 1/Gamma(r, r_s)
        comp33 = r**2
        comp44 = r**2 * np.sin(theta)**2

        Tensor = np.array([[-comp11, 0, 0, 0],
                           [ 0, comp22, 0, 0],
                           [ 0, 0, comp33, 0],
                           [ 0, 0, 0, comp44]], bytes)
        return Tensor
    
    def local_to_global_4momentum(vector_position, r_s, E, vect_napr_cont):
        # Вычисляем метрические коэффициенты
        gamma = 1 - r_s/vector_position[1]
        sqrt_gamma = np.sqrt(gamma)

        vect_napr_cov = np.zeros_like(vect_napr_cont)

        # присваиваем энергию
        vect_napr_cont[:, :, 0:3] *= E
        
        # Вычисляем контравариантные компоненты 4-импульса
        vect_napr_cont[:, :, 0] *= 1 / sqrt_gamma
        vect_napr_cont[:, :, 1] *= sqrt_gamma
        vect_napr_cont[:, :, 2] *= 1 / vector_position[1]
        vect_napr_cont[:, :, 3] *= 1 / (vector_position[1] * np.sin(vector_position[2]))
        
        # Преобразуем в ковариантные компоненты (для гамильтониана)
        vect_napr_cov[:, :, 0] = -gamma * vect_napr_cont[:, :, 0]               # g_{tt} = -gamma
        vect_napr_cov[:, :, 1] = vect_napr_cont[:, :, 1] / gamma                # g^{rr} = gamma
        vect_napr_cov[:, :, 2] = vector_position[1]**2 * vect_napr_cont[:, :, 2]                 # g_{\theta\theta} = r^2
        vect_napr_cov[:, :, 3] = (vector_position[1]*np.sin(vector_position[2]))**2 * vect_napr_cont[:, :, 3] # g_{\phi\phi} = (r sinθ)^2
        
        return vect_napr_cov
    
class VectorType(Enum):
    """Типы 4-векторов"""
    COORDINATES = auto()            # Вектор координат (t, x, y, z)
    IMPULSE = auto()                # Вектор импульса (E, px, py, pz)
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
    
        
class Camera:
    def __init__(self, width=10, height=10, focus=1.0):
        # Автоматическое преобразование в целые числа
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
        
        # Коэффициент пропорций сторон
        self.aspect_ratio = self.width / self.height

    def get_pixel_coordinates(self, i, j):
        """
        Преобразует пиксельные координаты (i, j) в нормализованные координаты камеры (u, v)
        
        Параметры:
            i: горизонтальная координата пикселя [0, width]
            j: вертикальная координата пикселя [0, height]
            
        Возвращает:
            (u, v): нормализованные координаты в пространстве камеры
        """
        # Преобразование в [-aspect_ratio, aspect_ratio] x [-1, 1]
        u = (2 * i - self.width) / self.width * self.aspect_ratio
        v = (2 * j - self.height) / self.height
        return u, v

    def create_camera_rays(self, camera_position, r_s, dtype=np.float64):
        """
        Создает массив начальных состояний лучей для камеры
        
        Параметры:
            camera_position: Vector4 - положение камеры в пространстве
            r_s: параметр метрики (радиус Шварцшильда)
            dtype: тип данных массива
            
        Возвращает:
            initial_states: массив формы (height, width, 8)
                где первые 4 компонента - координаты (t, x, y, z)
                последние 4 компонента - импульс (E, px, py, pz)
        """
        # Создаем сетку координат
        i_indices = np.arange(self.width)
        j_indices = np.arange(self.height)
        
        # Векторизованное вычисление координат (u, v)
        i_grid, j_grid = np.meshgrid(i_indices, j_indices)
        u_grid, v_grid = self.get_pixel_coordinates(i_grid, j_grid)
        
        # Фокусное расстояние (отрицательное, так как направлено "внутрь")
        f = -self.focus
        
        # Вычисляем норму пространственной части для каждого луча
        spatial_norm = np.sqrt(f**2 + v_grid**2 + u_grid**2)
        
        # Избегаем деления на ноль
        spatial_norm[spatial_norm == 0] = 1e-12
        
        # Создаем массив импульсов (E, px, py, pz)
        # Для фотона E = |p|, но здесь мы сохраняем направление
        impulse_array = np.zeros((self.height, self.width, 4), dtype=dtype)
        impulse_array[..., 0] = 1.0                         # Временная компонента (энергия)
        impulse_array[..., 1] = f / spatial_norm            # X-компонента
        impulse_array[..., 2] = v_grid / spatial_norm       # Y-компонента
        impulse_array[..., 3] = u_grid / spatial_norm       # Z-компонента
        
        # Преобразование импульсов в глобальную систему координат
        global_impulse = Metric.local_to_global_4momentum(
            camera_position.to_array(), 
            r_s, 
            1, 
            impulse_array
        )
        
        # Формирование начальных состояний: положение + импульс
        initial_states = np.zeros((self.height, self.width, 8), dtype=dtype)
        
        # Положение камеры одинаково для всех лучей
        cam_pos_array = camera_position.to_array()
        for i in range(4):
            initial_states[..., i] = cam_pos_array[i]
        
        # Импульсы
        initial_states[..., 4:] = global_impulse
        
        return initial_states


def plot_trajectories(trajectories, point_counts, r_s, W, H, step_ray=1):
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

    # Визуализация черной дыры
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_bh = r_s * np.outer(np.cos(u), np.sin(v))
    y_bh = r_s * np.outer(np.sin(u), np.sin(v))
    z_bh = r_s * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_bh, y_bh, z_bh, color='black', alpha=0.7, label='фактический горизонт событий')

    # Настройки визуализации
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Траектории света в метрике Шварцшильда\n(3D декартовы координаты)', fontsize=14)
    ax.legend()
    
    # Автомасштабирование
    all_points = trajectories[:, :, 1:4].reshape(-1, 3)
    valid_points = all_points[all_points[:, 0] > 1.001 * r_s]
    
    if valid_points.size > 0:
        max_val = np.max(valid_points)
        min_val = np.min(valid_points)
        max_range = max(np.abs(max_val), np.abs(min_val)) * 1.1
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

    # Фотонная сфера (круг в экваториальной плоскости)
    phi = np.linspace(0, 2*np.pi, 100)
    r_photon = 1.5 * r_s
    x_ph = r_photon * np.cos(phi)
    y_ph = r_photon * np.sin(phi)
    z_ph = np.zeros_like(phi)
    ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7, label='Фотонная сфера')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# @profile
def OpenCL(initial_states, lambda_0, lambda_end, h, r_s, global_size, max_points=1000, save_step=10):
    iterations = int( (lambda_end - lambda_0)/h )
    width, height = global_size
    total_rays = width * height

    # Инициализация OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Компиляция программы
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(script_dir, "OpenClKernel.cl")
    with open(kernel_path, encoding='utf-8') as f:
        kernel_code = f.read()
    prg = cl.Program(ctx, kernel_code).build()

    # Выделение памяти для результатов
    trajectories = np.zeros((total_rays, max_points, 8), dtype=np.float64)
    point_counts = np.zeros(total_rays, dtype=np.int32)

    # Размер батча
    batch_size = min(total_rays, 5000)  # Безопасный размер батча
    batches = (total_rays + batch_size - 1) // batch_size

    start_total = time.time()
    for batch_idx in range(batches):
        start_batch = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_rays)
        current_batch_size = end_idx - start_idx

        # Выделение памяти для батча
        mf = cl.mem_flags
        
        # Буфер начальных условий для батча
        initial_batch = initial_states[start_idx:end_idx]
        initial_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=initial_batch)
        
        # Буфер траекторий для батча
        trajectory_size_batch = current_batch_size * max_points
        trajectories_buf = cl.Buffer(ctx, mf.WRITE_ONLY, trajectory_size_batch * 8 * np.dtype(np.float64).itemsize)
        
        # Буфер счетчиков точек для батча
        point_counts_buf = cl.Buffer(ctx, mf.WRITE_ONLY, current_batch_size * np.dtype(np.int32).itemsize)
        
        # Запуск ядра для текущего батча
        prg.runge_kutta4_trajectories(
            queue, 
            (current_batch_size,),  # 1D NDRange
            None,
            np.float64(r_s),
            np.float64(h),
            np.int32(iterations),
            np.int32(max_points),
            np.int32(save_step),
            initial_buf,
            trajectories_buf,
            point_counts_buf
        )
        
        # Чтение результатов батча
        trajectories_batch_flat = np.empty(trajectory_size_batch * 8, dtype=np.float64)
        point_counts_batch = np.empty(current_batch_size, dtype=np.int32)
        
        # КОРОТКИЕ ИСПРАВЛЕНИЯ:
        cl.enqueue_copy(queue, trajectories_batch_flat, trajectories_buf)
        cl.enqueue_copy(queue, point_counts_batch, point_counts_buf)
        
        # Сохранение результатов
        trajectories_batch = trajectories_batch_flat.reshape(current_batch_size, max_points, 8)
        trajectories[start_idx:end_idx] = trajectories_batch
        point_counts[start_idx:end_idx] = point_counts_batch
        
        # Освобождаем ресурсы
        initial_buf.release()
        trajectories_buf.release()
        point_counts_buf.release()
        
        print(f"Батч {batch_idx+1}/{batches} ({current_batch_size} лучей) занял: {time.time() - start_batch:.2f} сек")
    
    print(f"Всего вычислено {total_rays} траекторий за {time.time() - start_total:.2f} секунд")
    
    return trajectories, point_counts


if __name__ == "__main__":
    # Параметры интегрирования
    r_s = 3.0           # Радиус Шварцшильда
    h = 0.01            # Шаг интегрирования
    lambda_0 = 0        # Начало интегрирования
    lambda_end = 200    # Конец интегрирования

    # Парметры камеры
    W = 10
    H = 10

    # Создаем масив начальных условий
    # (здесь должна быть ваша реализация Camera)
    camera_position = Vector4(t=0, x=20, y=np.pi/2, z=0, vtype=VectorType.COORDINATES, dtype=np.float32)      # позиция камеры в глобальных координатах

    camera = Camera(width=W, height=H, focus=2)
    initial_states = camera.create_camera_rays(camera_position, r_s, dtype=np.float64)
    initial_states_flat = initial_states.reshape(-1, 8)

    # Вычисляем траектории
    trajectories, point_counts = OpenCL(
        initial_states_flat,
        lambda_0,
        lambda_end,
        h,
        r_s,
        (W, H),
        max_points=500,
        save_step=10
    )

    # Визуализация
    plot_trajectories(trajectories, point_counts, r_s, W, H, step_ray=1)

# %%
