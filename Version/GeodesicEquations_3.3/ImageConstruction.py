import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Optional
from enum import Enum
import os

class RayStatus(Enum):
    """Статусы лучей согласно флагам из OpenCL ядра"""
    REACHED_SKY = 0      # Достиг небесной сферы
    FELL_INTO_BH = 1     # Упал в черную дыру
    ORBITING = 2         # Движется по орбите
    INVALID = 3          # Невалидное состояние

class PanoramicSkySphereRenderer:
    def __init__(self, point_status_2d: np.ndarray, point_flag_2d: np.ndarray, 
                 panorama_path: Optional[str] = None, panorama_image: Optional[np.ndarray] = None):
        """
        Инициализация рендерера для построения изображения на основе панорамной небесной сферы
        
        Args:
            point_status_2d: массив конечных состояний лучей в форме (Height, Width, 8)
            point_flag_2d: массив флагов состояния для каждого луча в форме (Height, Width)
            panorama_path: путь к панорамному изображению небесной сферы
            panorama_image: предзагруженное панорамное изображение (альтернатива panorama_path)
        """
        self.point_status = point_status_2d
        self.point_flag = point_flag_2d
        self.height, self.width = point_flag_2d.shape
        
        # Загрузка панорамного изображения
        if panorama_image is not None:
            self.panorama = panorama_image
        elif panorama_path is not None and os.path.exists(panorama_path):
            self.panorama = self._load_panorama(panorama_path)
        else:
            raise ValueError("Необходимо указать путь к панорамному изображению или предоставить само изображение")
            
        self.pano_height, self.pano_width, _ = self.panorama.shape
        
        # Цвета для различных статусов лучей (кроме REACHED_SKY, который берется из панорамы)
        self.status_colors = {
            RayStatus.FELL_INTO_BH: np.array([0, 0, 0]),        # Черный
            RayStatus.ORBITING: np.array([255, 0, 0]),          # Красный
            RayStatus.INVALID: np.array([0, 0, 255])            # Синий
        }
        
    def _load_panorama(self, path: str) -> np.ndarray:
        """Загрузка панорамного изображения"""
        image = Image.open(path)
        return np.array(image)
    
    def _normalize_angle(self, angle: float, period: float = 2*np.pi) -> float:
        """Нормализация угла к диапазону [0, period)"""
        return angle % period
    
    def _spherical_to_panorama_coords(self, theta: float, phi: float) -> Tuple[int, int]:
        """
        Преобразование сферических координат в координаты на панорамном изображении
        
        Args:
            theta: полярный угол [0, π]
            phi: азимутальный угол [0, 2π]
            
        Returns:
            Координаты (x, y) на панорамном изображении
        """
        # Нормализуем углы
        theta_norm = self._normalize_angle(theta, np.pi)
        phi_norm = self._normalize_angle(phi, 2*np.pi)
        
        # Преобразуем в координаты панорамы
        # x соответствует азимутальному углу (по ширине)
        x = int((phi_norm / (2 * np.pi)) * self.pano_width)
        
        # y соответствует полярному углу (по высоте)
        y = int((theta_norm / np.pi) * self.pano_height)
        
        # Обеспечиваем, чтобы координаты были в пределах изображения
        x = max(0, min(self.pano_width - 1, x))
        y = max(0, min(self.pano_height - 1, y))
        
        return x, y
    
    def render_image(self) -> np.ndarray:
        """
        Построение изображения на основе панорамной небесной сферы
        
        Returns:
            Изображение в формате (Height, Width, 3) с RGB значениями
        """
        # Создаем пустое изображение
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Обрабатываем каждый пиксель изображения
        for i in range(self.height):
            for j in range(self.width):
                status_value = self.point_flag[i, j]
                status = RayStatus(status_value)
                
                if status == RayStatus.REACHED_SKY:
                    # Для лучей, достигших небесной сферы, берем цвет из панорамы
                    state = self.point_status[i, j]
                    theta = state[2]  # Полярный угол
                    phi = state[3]    # Азимутальный угол
                    
                    # Преобразуем углы в координаты панорамы
                    x, y = self._spherical_to_panorama_coords(theta, phi)
                    
                    # Берем цвет из панорамного изображения
                    image[i, j] = self.panorama[y, x]
                else:
                    # Для других статусов используем предопределенные цвета
                    image[i, j] = self.status_colors[status]
        
        return image
    
    def render_with_intensity(self, intensity_factor: float = 1.0) -> np.ndarray:
        """
        Построение изображения с учетом интенсивности (энергии) фотонов
        
        Args:
            intensity_factor: коэффициент усиления/ослабления интенсивности
            
        Returns:
            Изображение в формате (Height, Width, 3) с RGB значениями
        """
        # Создаем пустое изображение
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Обрабатываем каждый пиксель изображения
        for i in range(self.height):
            for j in range(self.width):
                status_value = self.point_flag[i, j]
                status = RayStatus(status_value)
                
                if status == RayStatus.REACHED_SKY:
                    # Для лучей, достигших небесной сферы, берем цвет из панорамы
                    state = self.point_status[i, j]
                    theta = state[2]  # Полярный угол
                    phi = state[3]    # Азимутальный угол
                    
                    # Преобразуем углы в координаты панорамы
                    x, y = self._spherical_to_panorama_coords(theta, phi)
                    
                    # Берем цвет из панорамного изображения
                    color = self.panorama[y, x].astype(float)
                    
                    # Учитываем интенсивность (энергию фотона)
                    energy = abs(state[4])  # p_t компонента (энергия)
                    color *= energy * intensity_factor
                    
                    # Ограничиваем значения и преобразуем обратно в uint8
                    color = np.clip(color, 0, 255).astype(np.uint8)
                    image[i, j] = color
                else:
                    # Для других статусов используем предопределенные цвета
                    image[i, j] = self.status_colors[status]
        
        return image
    
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
            RayStatus.INVALID: 0
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
        img = Image.fromarray(image)
        img.save(path)