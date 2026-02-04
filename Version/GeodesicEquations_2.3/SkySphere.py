import numpy as np
from PIL import Image
import math
import os

class SkySphere:
    def __init__(self, image_path):
        """
        Инициализирует объект небесной сферы с панорамным изображением.
        
        Args:
            image_path (str): Путь к панорамному изображению (абсолютный или относительный)
        """
        # Получаем абсолютный путь к файлу
        if not os.path.isabs(image_path):
            # Если путь относительный, преобразуем его в абсолютный
            script_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(script_dir, image_path)
        
        # Проверяем существование файла
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл изображения не найден: {image_path}")
        
        # Загружаем изображение
        try:
            self.image = Image.open(image_path)
            self.width, self.height = self.image.size
            self.pixels = self.image.load()
            
            # Предварительно конвертируем в массив numpy для более быстрого доступа
            self.image_array = np.array(self.image)
        except Exception as e:
            raise ValueError(f"Ошибка загрузки изображения: {e}")
    
    def get_color(self, theta, phi, interpolate=True):
        """
        Возвращает цвет пикселя на сфере для заданных сферических координат.
        
        Args:
            theta (float): Полярный угол (от 0 до π)
            phi (float): Азимутальный угол (от 0 до 2π)
            interpolate (bool): Использовать ли билинейную интерполяцию
            
        Returns:
            tuple: Цвет в формате (R, G, B)
        """
        # Преобразуем сферические координаты в координаты изображения
        # phi: 0-2π -> x: 0-width
        # theta: 0-π -> y: 0-height
        
        # Нормализуем углы
        phi = phi % (2 * math.pi)
        theta = max(0, min(math.pi, theta))
        
        # Вычисляем координаты на изображении
        x = (phi / (2 * math.pi)) * self.width
        y = (theta / math.pi) * self.height
        
        if interpolate:
            return self._bilinear_interpolation(x, y)
        else:
            # Без интерполяции - просто берем ближайший пиксель
            x_idx = min(self.width - 1, max(0, int(round(x))))
            y_idx = min(self.height - 1, max(0, int(round(y))))
            return self.pixels[x_idx, y_idx]
    
    def _bilinear_interpolation(self, x, y):
        """
        Выполняет билинейную интерполяцию для получения цвета.
        
        Args:
            x (float): Координата X на изображении
            y (float): Координата Y на изображении
            
        Returns:
            tuple: Интерполированный цвет (R, G, B)
        """
        # Находим четыре ближайших пикселя
        x0 = int(math.floor(x))
        x1 = min(self.width - 1, x0 + 1)
        y0 = int(math.floor(y))
        y1 = min(self.height - 1, y0 + 1)
        
        # Вычисляем веса для интерполяции
        x_ratio = x - x0
        y_ratio = y - y0
        x_opposite = 1 - x_ratio
        y_opposite = 1 - y_ratio
        
        # Получаем цвета четырех пикселей
        try:
            c00 = self.pixels[x0, y0]
            c01 = self.pixels[x0, y1]
            c10 = self.pixels[x1, y0]
            c11 = self.pixels[x1, y1]
        except IndexError:
            # Если вышли за границы, возвращаем черный цвет
            return (0, 0, 0)
        
        # Интерполируем по горизонтали
        c0 = tuple(
            c00[i] * x_opposite + c10[i] * x_ratio
            for i in range(len(c00))
        )
        c1 = tuple(
            c01[i] * x_opposite + c11[i] * x_ratio
            for i in range(len(c01))
        )
        
        # Интерполируем по вертикали
        result = tuple(
            int(c0[i] * y_opposite + c1[i] * y_ratio)
            for i in range(len(c0))
        )
        
        return result
    
    def get_color_cartesian(self, x, y, z, interpolate=True):
        """
        Возвращает цвет для точки на сфере, заданной в декартовых координатах.
        
        Args:
            x, y, z (float): Координаты точки на единичной сфере
            interpolate (bool): Использовать ли билинейную интерполяцию
            
        Returns:
            tuple: Цвет в формате (R, G, B)
        """
        # Нормализуем вектор
        length = math.sqrt(x*x + y*y + z*z)
        if length == 0:
            return (0, 0, 0)  # Нулевой вектор - черный цвет
        
        x /= length
        y /= length
        z /= length
        
        # Преобразуем декартовы координаты в сферические
        theta = math.acos(z)  # Полярный угол (0 до π)
        phi = math.atan2(y, x)  # Азимутальный угол (-π до π)
        
        # Приводим φ к диапазону 0-2π
        if phi < 0:
            phi += 2 * math.pi
        
        return self.get_color(theta, phi, interpolate)
    
    def get_direction_color(self, direction, interpolate=True):
        """
        Возвращает цвет для заданного направления (нормализованного вектора).
        
        Args:
            direction (tuple/list): Нормализованный вектор направления (x, y, z)
            interpolate (bool): Использовать ли билинейную интерполяцию
            
        Returns:
            tuple: Цвет в формате (R, G, B)
        """
        return self.get_color_cartesian(direction[0], direction[1], direction[2], interpolate)

class SkyProjection:
    def __init__(self, sky_sphere, width, height):
        """
        Оптимизированная версия с использованием векторизации NumPy.
        """
        self.sky_sphere = sky_sphere
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
    
    def create_projection(self, directions):
        """
        Создает проекцию на основе массива направлений (векторизованная версия).
        """
        if directions.shape[:2] != (self.height, self.width):
            raise ValueError("Размер массива направлений не соответствует размеру изображения")
        
        # Векторизованное получение цветов
        theta_flat = directions[:, :, 0].flatten()
        phi_flat = directions[:, :, 1].flatten()
        
        # Получаем цвета для всех направлений
        colors_flat = np.zeros((len(theta_flat), 3), dtype=np.uint8)
        for idx, (theta, phi) in enumerate(zip(theta_flat, phi_flat)):
            colors_flat[idx] = self.sky_sphere.get_color(theta, phi)
        
        # Преобразуем обратно в форму изображения
        self.image = colors_flat.reshape(self.height, self.width, 3)
        
        return Image.fromarray(self.image)
    
    def create_equirectangular_projection(self):
        """
        Создает эквидистантную проекцию (векторизованная версия).
        """
        # Создаем сетку координат
        i_coords, j_coords = np.mgrid[0:self.height, 0:self.width]
        
        # Вычисляем theta и phi для каждой точки
        theta = (i_coords / self.height) * math.pi
        phi = (j_coords / self.width) * 2 * math.pi
        
        # Объединяем в массив направлений
        directions = np.stack([theta, phi], axis=-1)
        
        return self.create_projection(directions)
    
    def create_fisheye_projection(self, fov=math.pi):
        """
        Создает проекцию "рыбий глаз" (векторизованная версия).
        """
        # Создаем сетку координат
        i_coords, j_coords = np.mgrid[0:self.height, 0:self.width]
        
        center_x = self.width / 2
        center_y = self.height / 2
        max_radius = min(center_x, center_y)
        
        # Вычисляем расстояния и углы
        dx = j_coords - center_x
        dy = i_coords - center_y
        r = np.sqrt(dx*dx + dy*dy) / max_radius
        
        # Маска для пикселей внутри круга
        mask = r <= 1.0
        
        # Вычисляем углы
        phi = np.arctan2(dy, dx)
        phi[phi < 0] += 2 * math.pi
        
        # Вычисляем theta
        theta = r * fov / 2
        
        # За пределами круга - черный цвет
        theta[~mask] = 0
        phi[~mask] = 0
        
        # Объединяем в массив направлений
        directions = np.stack([theta, phi], axis=-1)
        
        return self.create_projection(directions)
    
    def save_image(self, filename):
        """
        Сохраняет изображение в файл.
        """
        Image.fromarray(self.image).save(filename)


# Пример использования с обработкой ошибок
if __name__ == "__main__":
    # Создаем пользовательский массив направлений
    height, width = 400, 600
    directions = np.zeros((height, width, 2))

    # Заполняем массив направлений
    for i in range(height):
        for j in range(width):
            # Ваша логика вычисления направлений
            theta = (i / height)/2 * math.pi
            phi = (j / width)/2 * 2 * math.pi
            
            directions[i, j] = [theta, phi]

    # Создаем проекцию
    sky_sphere = SkySphere("rogland_clear_night_4k.png")
    projector = SkyProjection(sky_sphere, width, height)
    result_image = projector.create_projection(directions)
    result_image.save("projection.png")
