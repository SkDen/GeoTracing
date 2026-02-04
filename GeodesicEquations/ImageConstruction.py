import os
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
from typing import Tuple, Optional, Dict, Union, List
from enum import Enum
from tqdm import tqdm

from scipy import ndimage
from scipy.interpolate import griddata

class InterpolationMethod(Enum):
    """Методы интерполяции для улучшения качества изображения"""
    NEAREST = "nearest"        # Ближайший сосед (быстро, но низкое качество)
    BILINEAR = "bilinear"      # Биллинейная интерполяция
    BICUBIC = "bicubic"        # Бикубическая интерполяция
    LANCZOS = "lanczos"        # Интерполяция Ланцоша (высокое качество)
    GAUSSIAN = "gaussian"      # Гауссово сглаживание
    ADAPTIVE = "adaptive"      # Адаптивная интерполяция на основе энергии

class RayStatus(Enum):
    """Статусы лучей согласно флагам из OpenCL ядра"""
    REACHED_SKY = 0      # Достиг небесной сферы
    FELL_INTO_BH = 1     # Упал в черную дыру
    ORBITING = 2         # Движется по орбите
    INVALID = 3          # Невалидное состояние
    ANOTHER_UNIVERSE = 4 # Небесная сфера в другой вселенной

class PanoramicSkySphereRenderer:
    def __init__(self, point_status_2d: np.ndarray, point_flag_2d: np.ndarray, 
                 panorama_path: Optional[str] = None, 
                 panorama_another_path: Optional[str] = None,
                 panorama_image: Optional[np.ndarray] = None,
                 panorama_another_image: Optional[np.ndarray] = None,
                 interpolation_method: InterpolationMethod = InterpolationMethod.BILINEAR):
        """
        Инициализация рендерера для построения изображения на основе панорамной небесной сферы
        
        Args:
            point_status_2d: массив конечных состояний лучей в форме (Height, Width, 8)
            point_flag_2d: массив флагов состояния для каждого луча в форме (Height, Width)
            panorama_path: путь к панорамному изображению основной небесной сферы
            panorama_another_path: путь к панорамному изображению другой вселенной
            panorama_image: предзагруженное панорамное изображение (альтернатива panorama_path)
            panorama_another_image: предзагруженное панорамное изображение другой вселенной
            interpolation_method: метод интерполяции для улучшения качества изображения
        """
        self.point_status = point_status_2d
        self.point_flag = point_flag_2d
        self.height, self.width = point_flag_2d.shape
        self.interpolation_method = interpolation_method
        
        # Загрузка панорамных изображений
        self.panoramas = {}
        
        # Основная панорама
        if panorama_image is not None:
            self.panoramas[RayStatus.REACHED_SKY] = panorama_image
        elif panorama_path is not None and os.path.exists(panorama_path):
            self.panoramas[RayStatus.REACHED_SKY] = self._load_panorama(panorama_path)
        else:
            raise ValueError("Необходимо указать путь к основному панорамному изображению или предоставить само изображение")
            
        # Панорама для другой вселенной
        if panorama_another_image is not None:
            self.panoramas[RayStatus.ANOTHER_UNIVERSE] = panorama_another_image
        elif panorama_another_path is not None and os.path.exists(panorama_another_path):
            self.panoramas[RayStatus.ANOTHER_UNIVERSE] = self._load_panorama(panorama_another_path)
        else:
            # Если не указана отдельная панорама для другой вселенной, используем основную
            self.panoramas[RayStatus.ANOTHER_UNIVERSE] = self.panoramas[RayStatus.REACHED_SKY]
            
        # Размеры панорам (предполагаем, что все панорамы имеют одинаковый размер)
        self.pano_height, self.pano_width, _ = self.panoramas[RayStatus.REACHED_SKY].shape
        
        # Цвета для различных статусов лучей (кроме REACHED_SKY и ANOTHER_UNIVERSE, которые берутся из панорамы)
        self.status_colors = {
            RayStatus.FELL_INTO_BH: np.array([0, 0, 0]),        # Черный
            RayStatus.ORBITING: np.array([255, 0, 0]),          # Красный
            RayStatus.INVALID: np.array([0, 0, 255])            # Синий
        }
        
    def _load_panorama(self, path: str) -> np.ndarray:
        """Загрузка панорамного изображения"""
        image = Image.open(path)
        # Конвертируем в RGB если нужно
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    
    def _normalize_angle(self, angle: float, period: float = 2*np.pi) -> float:
        """Нормализация угла к диапазону [0, period)"""
        return angle % period
    
    def _spherical_to_panorama_coords(self, theta: float, phi: float) -> Tuple[float, float]:
        """
        Преобразование сферических координат в координаты на панорамном изображении
        
        Args:
            theta: полярный угол [0, π]
            phi: азимутальный угол [0, 2π]
            
        Returns:
            Координаты (x, y) на панорамном изображении (дробные)
        """
        # Нормализуем углы
        theta_norm = self._normalize_angle(theta, np.pi)
        phi_norm = self._normalize_angle(phi, 2*np.pi)
        
        # Преобразуем в координаты панорамы
        # x соответствует азимутальному углу (по ширине)
        x = (phi_norm / (2 * np.pi)) * (self.pano_width - 1)
        
        # y соответствует полярному углу (по высоте)
        y = (theta_norm / np.pi) * (self.pano_height - 1)
        
        return x, y
    
    def _get_pixel_interpolated(self, x: float, y: float, panorama: np.ndarray, 
                               method: InterpolationMethod = None) -> np.ndarray:
        """
        Получение интерполированного значения пикселя
        
        Args:
            x: координата X (дробная)
            y: координата Y (дробная)
            panorama: панорамное изображение для интерполяции
            method: метод интерполяции (если None, используется self.interpolation_method)
            
        Returns:
            Интерполированный цвет пикселя
        """
        if method is None:
            method = self.interpolation_method
            
        # Обеспечиваем, чтобы координаты были в пределах изображения
        x = max(0, min(self.pano_width - 1, x))
        y = max(0, min(self.pano_height - 1, y))
        
        if method == InterpolationMethod.NEAREST:
            # Ближайший сосед (быстро, но низкое качество)
            return panorama[int(round(y)), int(round(x))]
        
        elif method == InterpolationMethod.BILINEAR:
            # Биллинейная интерполяция
            x_floor, y_floor = int(np.floor(x)), int(np.floor(y))
            x_ceil, y_ceil = min(x_floor + 1, self.pano_width - 1), min(y_floor + 1, self.pano_height - 1)
            
            # Коэффициенты интерполяции
            x_ratio = x - x_floor
            y_ratio = y - y_floor
            x_opposite = 1 - x_ratio
            y_opposite = 1 - y_ratio
            
            # Интерполяция по каждому каналу
            result = np.zeros(3)
            for c in range(3):
                result[c] = (
                    panorama[y_floor, x_floor, c] * x_opposite * y_opposite +
                    panorama[y_floor, x_ceil, c] * x_ratio * y_opposite +
                    panorama[y_ceil, x_floor, c] * x_opposite * y_ratio +
                    panorama[y_ceil, x_ceil, c] * x_ratio * y_ratio
                )
            
            return result.astype(np.uint8)
        
        elif method == InterpolationMethod.BICUBIC:
            # Бикубическая интерполяция с использованием scipy
            # Для каждого канала отдельно
            result = np.zeros(3)
            for c in range(3):
                result[c] = ndimage.map_coordinates(
                    panorama[:, :, c], 
                    [[y], [x]], 
                    order=3, 
                    mode='nearest'
                )
            return result.astype(np.uint8)
        
        elif method == InterpolationMethod.LANCZOS:
            # Интерполяция Ланцоша (высокое качество)
            # Для каждого канала отдельно
            result = np.zeros(3)
            for c in range(3):
                result[c] = ndimage.map_coordinates(
                    panorama[:, :, c], 
                    [[y], [x]], 
                    order=3, 
                    mode='nearest',
                    prefilter=False
                )
            return result.astype(np.uint8)
        
        else:
            # По умолчанию используем билинейную интерполяцию
            return self._get_pixel_interpolated(x, y, panorama, InterpolationMethod.BILINEAR)
    
    def _apply_adaptive_smoothing(self, image: np.ndarray, energy_map: np.ndarray) -> np.ndarray:
        """
        Применение адаптивного сглаживания на основе энергии фотонов
        
        Args:
            image: исходное изображение
            energy_map: карта энергии фотонов
            
        Returns:
            Сглаженное изображение
        """
        # Нормализуем энергию
        energy_norm = energy_map / np.max(energy_map) if np.max(energy_map) > 0 else energy_map
        
        # Применяем адаптивное сглаживание
        # Области с низкой энергией сглаживаем сильнее
        result = np.zeros_like(image, dtype=float)
        
        for i in range(self.height):
            for j in range(self.width):
                # Определяем размер ядра сглаживания на основе энергии
                kernel_size = max(1, int(5 * (1 - energy_norm[i, j])))
                
                # Определяем область для сглаживания
                i_min = max(0, i - kernel_size)
                i_max = min(self.height, i + kernel_size + 1)
                j_min = max(0, j - kernel_size)
                j_max = min(self.width, j + kernel_size + 1)
                
                # Взвешенное среднее с учетом энергии
                weights = energy_norm[i_min:i_max, j_min:j_max]
                weights_sum = np.sum(weights)
                
                if weights_sum > 0:
                    for c in range(3):
                        result[i, j, c] = np.sum(
                            image[i_min:i_max, j_min:j_max, c] * weights
                        ) / weights_sum
                else:
                    result[i, j] = image[i, j]
        
        return result.astype(np.uint8)
    
    def render_image(self, use_interpolation: bool = True,
                     show_progress: bool = True,
                     rotation_sphere_phi: float = 0) -> np.ndarray:
        """
        Построение изображения на основе панорамной небесной сферы
        
        Args:
            use_interpolation: использовать ли интерполяцию для улучшения качества
            
        Returns:
            Изображение в формате (Height, Width, 3) с RGB значениями
        """
        # Создаем пустое изображение
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Создаем карту энергии для адаптивного сглаживания
        energy_map = np.zeros((self.height, self.width))

        self.point_status[:, :, 3] += rotation_sphere_phi

        # Инициализируем прогресс-бар
        rows_range = range(self.height)
        if show_progress:
            rows_range = tqdm(rows_range, desc="Построение изображения: ", unit=" строк")
        
        # Обрабатываем каждый пиксель изображения
        for i in rows_range:
            for j in range(self.width):
                status_value = self.point_flag[i, j]
                status = RayStatus(status_value)
                
                if status == RayStatus.REACHED_SKY or status == RayStatus.ANOTHER_UNIVERSE:
                    # Для лучей, достигших небесной сферы, берем цвет из соответствующей панорамы
                    state = self.point_status[i, j]
                    theta = state[2]  # Полярный угол
                    phi = state[3]    # Азимутальный угол
                    
                    # Преобразуем углы в координаты панорамы
                    x, y = self._spherical_to_panorama_coords(theta, phi)
                    
                    # Берем цвет из соответствующего панорамного изображения
                    panorama = self.panoramas[status]
                    if use_interpolation:
                        image[i, j] = self._get_pixel_interpolated(x, y, panorama)
                    else:
                        # Без интерполяции - ближайший сосед
                        x_int = int(round(x))
                        y_int = int(round(y))
                        image[i, j] = panorama[y_int, x_int]
                    
                    # Сохраняем энергию для адаптивного сглаживания
                    energy_map[i, j] = abs(state[4])  # p_t компонента (энергия)
                else:
                    # Для других статусов используем предопределенные цвета
                    image[i, j] = self.status_colors[status]
        
        # Применяем адаптивное сглаживание если нужно
        if use_interpolation and self.interpolation_method == InterpolationMethod.ADAPTIVE:
            image = self._apply_adaptive_smoothing(image, energy_map)
        
        return image
    
    def render_with_intensity(self, intensity_factor: float = 1.0, 
                             use_interpolation: bool = True,
                             show_progress: bool = True,
                             rotation_sphere_phi: float = 0) -> np.ndarray:
        """
        Построение изображения с учетом интенсивности (энергии) фотонов
        
        Args:
            intensity_factor: коэффициент усиления/ослабления интенсивности
            use_interpolation: использовать ли интерполяцию для улучшения качества
            
        Returns:
            Изображение в формате (Height, Width, 3) с RGB значениями
        """
        # Создаем пустое изображение
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Создаем карту энергии для адаптивного сглаживания
        energy_map = np.zeros((self.height, self.width))

        self.point_status[:, :, 3] += rotation_sphere_phi

        # Инициализируем прогресс-бар
        rows_range = range(self.height)
        if show_progress:
            rows_range = tqdm(rows_range, desc="Построение изображения: ", unit=" строк")
        
        # Обрабатываем каждый пиксель изображения
        for i in rows_range:
            for j in range(self.width):
                status_value = self.point_flag[i, j]
                status = RayStatus(status_value)
                
                if status == RayStatus.REACHED_SKY or status == RayStatus.ANOTHER_UNIVERSE:
                    # Для лучей, достигших небесной сферы, берем цвет из соответствующей панорамы
                    state = self.point_status[i, j]
                    theta = state[2]  # Полярный угол
                    phi = state[3]    # Азимутальный угол
                    
                    # Преобразуем углы в координаты панорамы
                    x, y = self._spherical_to_panorama_coords(theta, phi)
                    
                    # Берем цвет из соответствующего панорамного изображения
                    panorama = self.panoramas[status]
                    if use_interpolation:
                        color = self._get_pixel_interpolated(x, y, panorama).astype(float)
                    else:
                        # Без интерполяции - ближайший сосед
                        x_int = int(round(x))
                        y_int = int(round(y))
                        color = panorama[y_int, x_int].astype(float)
                    
                    # Учитываем интенсивность (энергию фотона)
                    energy = abs(state[4])  # p_t компонента (энергия)
                    energy_map[i, j] = energy
                    
                    color *= energy * intensity_factor
                    
                    # Ограничиваем значения и преобразуем обратно в uint8
                    color = np.clip(color, 0, 255).astype(np.uint8)
                    image[i, j] = color
                else:
                    # Для других статусов используем предопределенные цвета
                    image[i, j] = self.status_colors[status]
        
        # Применяем адаптивное сглаживание если нужно
        if use_interpolation and self.interpolation_method == InterpolationMethod.ADAPTIVE:
            image = self._apply_adaptive_smoothing(image, energy_map)
        
        return image
    
    # Остальные методы остаются без изменений
    def render_with_gradient_compensation(self, gradient_strength: float = 0.5,
                                          rotation_sphere_phi: float = 0) -> np.ndarray:
        """
        Построение изображения с компенсацией градиента для сглаживания резких переходов
        
        Args:
            gradient_strength: сила компенсации градиента (0-1)
            
        Returns:
            Изображение в формате (Height, Width, 3) с RGB значениями
        """
        # Сначала рендерим обычное изображение
        base_image = self.render_image(use_interpolation=True, rotation_sphere_phi=rotation_sphere_phi)
        
        # Вычисляем градиент изображения
        gradient_x = ndimage.sobel(base_image, axis=1)
        gradient_y = ndimage.sobel(base_image, axis=0)
        
        # Комбинируем градиенты
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Нормализуем градиент
        if np.max(gradient_magnitude) > 0:
            gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
        
        # Применяем размытие для сглаживания градиента
        blurred_image = ndimage.gaussian_filter(base_image, sigma=1)
        
        # Смешиваем исходное изображение с размытым на основе градиента
        # В областях с высоким градиентом используем больше размытого изображения
        result = np.zeros_like(base_image, dtype=float)
        
        for i in range(self.height):
            for j in range(self.width):
                # Коэффициент смешивания на основе градиента
                blend_factor = gradient_magnitude[i, j] * gradient_strength
                
                # Смешиваем изображения
                result[i, j] = (1 - blend_factor) * base_image[i, j] + blend_factor * blurred_image[i, j]
        
        return result.astype(np.uint8)
    
    def visualize_comparison(self, save_path: Optional[str] = None):
        """
        Визуализация сравнения различных методов интерполяции
        
        Args:
            save_path: путь для сохранения изображения сравнения
        """
        # Создаем изображения с разными методами
        images = {
            "Без интерполяции": self.render_image(use_interpolation=False),
            "Билинейная": self.render_image(use_interpolation=True),
            "С градиентной компенсацией": self.render_with_gradient_compensation(),
            "С интенсивностью": self.render_with_intensity()
        }
        
        # Создаем фигуру для сравнения
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (title, image) in enumerate(images.items()):
            axes[idx].imshow(image)
            axes[idx].set_title(title)
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Сохраняем изображение если указан путь
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def visualize(self, image: Optional[np.ndarray] = None, 
                  save_path: Optional[str] = None, 
                  title: str = "Rendered Image"):
        """
        Визуализация результатов
        
        Args:
            image: изображение для визуализации (если None, будет построено)
            save_path: путь для сохранения изображения
            title: заголовок изображения
        """
        if image is None:
            image = self.render_image()
        
        # Создаем фигуру
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        
        # Сохраняем изображение если указан путь
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def get_statistics(self) -> dict:
        """
        Получение статистики по результатам трассировки
        
        Returns:
            Словарь со статистикой
        """
        total_rays = self.height * self.width
        status_counts = {
            RayStatus.REACHED_SKY: 0,
            RayStatus.FELL_INTO_BH: 0,
            RayStatus.ORBITING: 0,
            RayStatus.INVALID: 0,
            RayStatus.ANOTHER_UNIVERSE: 0
        }
        
        # Подсчитываем количество лучей каждого статуса
        for i in range(self.height):
            for j in range(self.width):
                status_value = self.point_flag[i, j]
                status = RayStatus(status_value)
                status_counts[status] += 1
        
        # Вычисляем проценты
        status_percentages = {
            status.name: (count / total_rays * 100)
            for status, count in status_counts.items()
        }
        
        return {
            "total_rays": total_rays,
            "status_counts": status_counts,
            "status_percentages": status_percentages
        }
    
    def save_image(self, image: np.ndarray, path: str):
        """
        Сохранение изображения в файл
        
        Args:
            image: изображение для сохранения
            path: путь для сохранения
        """
        # Преобразуем массив в изображение PIL и сохраняем
        flipped_image = Image.fromarray(image)
        flipped_image.save(path)