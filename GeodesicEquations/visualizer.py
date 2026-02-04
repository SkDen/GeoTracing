import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from typing import List, Tuple, Optional, Callable
from enums import MetricType, CoordinatesType


class TrajectoryVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Инициализация визуализатора траекторий
        
        Args:
            figsize: размер фигуры matplotlib
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.trajectory_count = 0
        self.color_map = plt.cm.viridis
        
        # Параметры по умолчанию
        self.default_params = {
            'r_s': 1.0,
            'r_0': 1.0,
            'M': 1.0,
            'Q': 1.0,
            'L': 1.0,
            'scale_factor': 1.0,
            'static_scale_factor': 1.0,
            'hubble_parameter': 1.0
        }
        
    def set_parameters(self, **kwargs):
        """Установка параметров метрики"""
        for key, value in kwargs.items():
            if key in self.default_params:
                self.default_params[key] = value
                
    def _spherical_to_cartesian(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Преобразование сферических координат в декартовы"""
        theta = np.where(np.isclose(theta, 0), 1e-10, theta)
        theta = np.where(np.isclose(theta, np.pi), np.pi - 1e-10, theta)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z
    
    def _cylindrical_to_cartesian(self, r: np.ndarray, phi: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Преобразование цилиндрических координат в декартовы"""
        phi = np.where(np.isclose(r, 0), 0, phi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y, z
    
    def _specal_cylindical(self, R: np.ndarray, nu: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ортогональная проекция 4-х мерных цилиндрических координат"""
        x = R*np.sin(theta)*np.cos(phi)
        y = R*np.sin(theta)*np.sin(phi)
        z = nu
        return x, y, z
    
    def _hyperspheric_to_cartesian_stereographic(self, t: np.ndarray, chi: np.ndarray, 
                                               theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Преобразование гиперсферических координат в декартовы (стереографическая проекция)"""
        chi = np.clip(chi, 1e-10, 2*np.pi - 1e-10)
        
        a = self.default_params['static_scale_factor'] + np.exp(self.default_params['hubble_parameter'] * t)
        cot = np.cos(chi/2) / np.sin(chi/2)
        
        x = a * cot * np.sin(theta) * np.cos(phi)
        y = a * cot * np.sin(theta) * np.sin(phi)
        z = a * cot * np.cos(theta)

        return x, y, z
    
    def _hyperspheric_to_cartesian_orthogonal(self, t: np.ndarray, chi: np.ndarray, 
                                            theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Преобразование гиперсферических координат в декартовы (ортогональная проекция)"""
        a = self.default_params['static_scale_factor'] + np.exp(self.default_params['hubble_parameter'] * t)
        
        sin_chi = np.sin(chi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        x = a * sin_chi * sin_theta * cos_phi
        y = a * sin_chi * sin_theta * sin_phi
        z = a * sin_chi * cos_theta

        return x, y, z
    
    def _draw_reference_grid(self, chi: float, conversion_func: Callable, ax=None):
        """Отрисовка справочной сетки для гиперсферических координат"""
        if ax is None:
            ax = self.ax

        theta_mass = np.linspace(1e-10, np.pi, 10)
        for theta in theta_mass:
            phi = np.linspace(1e-10, 2*np.pi, 100)
            x, y, z = conversion_func(1, chi, theta, phi)
            ax.plot(x, y, z, linewidth=0.7, color=(1.0, 0.0, 0.0, 0.5))

        phi_mass = np.linspace(1e-10, 2*np.pi, 10)
        for phi in phi_mass:
            theta = np.linspace(1e-10, 2*np.pi, 100)
            x, y, z = conversion_func(1, chi, theta, phi)
            ax.plot(x, y, z, linewidth=0.7, color=(0.0, 1.0, 0.0, 0.5))

    def _draw_reference_special_cylindrical_grid(self, ax=None):
        """Отрисовка справочной сетки для специальных цилиндрических координат"""
        if ax is None:
            ax = self.ax

        theta = np.pi/2

        phi_mass = np.linspace(1e-10, 2*np.pi, 10)
        for phi in phi_mass:
            nu = np.linspace(-20, 20, 5)
            x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
            ax.plot(x, y, z, linewidth=0.7, color=(1.0, 0.0, 0.0, 0.7))

        nu_mass = np.linspace(-20, 20, 10)
        for nu in nu_mass:
            phi = np.linspace(1e-10, 2*np.pi, 50)
            x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
            ax.plot(x, y, z, linewidth=0.7, color=(1.0, 0.0, 0.0, 0.7))     
    
    def _draw_surface(self, r, color: str, alpha: float, num: int, name: str):
        """
        Построение поверхности
        
        Args:
            r: радиус или функция, возвращающая радиус
            color: цвет поверхности
            alpha: прозрачность
            num: количество точек для построения
            name: название поверхности для легенды
        """
        phi = np.linspace(0, 2 * np.pi, num)
        theta = np.linspace(0, np.pi, num)
        phi, theta = np.meshgrid(phi, theta)

        # Проверяем, является ли параметр функцией или переменной
        if callable(r):
            R = r(theta, phi)
        else:
            R = r

        X = R * np.sin(theta) * np.cos(phi)
        Y = R * np.sin(theta) * np.sin(phi)
        Z = R * np.cos(theta)

        surf = self.ax.plot_surface(X, Y, Z, color=color, alpha=alpha, label=name)
        return surf
    
    def _draw_schwarzschild_surfaces(self, ax=None):
        """Отрисовка поверхностей для метрики Шварцшильда"""
        if ax is None:
            ax = self.ax
        
        # Горизонт событий
        self._draw_surface(self.default_params['r_s'], 'black', 0.7, 50, 'Горизонт событий')
        
        # Фотонная сфера
        phi = np.linspace(0, 2*np.pi, 100)
        r_photon = 1.5 * self.default_params['r_s']
        
        # Фотонная сфера (круг в экваториальной плоскости)
        x_ph = r_photon * np.cos(phi)
        y_ph = r_photon * np.sin(phi)
        z_ph = np.zeros_like(phi)
        self.ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7, label='Фотонная сфера')
        
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
        
        ax.set_title('Траектории света в метрике Шварцшильда\n(3D декартовы координаты)', fontsize=14)
    
    def _draw_ellis_bronnikova_surfaces(self, ax=None):
        """Отрисовка поверхностей для метрики Эллиса-Бронникова"""
        if ax is None:
            ax = self.ax

        self._draw_surface(self.default_params['r_0'], 'black', 0.7, 50, 'Горловина червоточины')
        ax.set_title('Траектории света в метрике Эллиса-Бронникова\n(3D декартовы координаты)', fontsize=14)
    
    def _draw_kerr_newman_surfaces(self, ax=None):
        """Отрисовка поверхностей для метрики Керра-Ньюмена"""
        if ax is None:
            ax = self.ax

        M = self.default_params['M']
        Q = self.default_params['Q']
        L = self.default_params['L']
        
        # Радиус эргосфер
        def r_erg_pl(theta, phi):
            param = M**2 - Q**2 - (L/M)**2 * np.cos(theta)**2
            return np.where(param >= 0, M + np.sqrt(param), np.nan)
        
        def r_erg_in(theta, phi):
            param = M**2 - Q**2 - (L/M)**2 * np.cos(theta)**2
            return np.where(param >= 0, M - np.sqrt(param), np.nan)
        
        # Радиусы горизонтов
        def r_pl(theta, phi):
            param = M**2 - Q**2 - (L/M)**2
            r_val = M + np.sqrt(param) if param >= 0 else np.nan
            return np.full_like(theta, r_val)
        
        def r_in(theta, phi):
            param = M**2 - Q**2 - (L/M)**2
            r_val = M - np.sqrt(param) if param >= 0 else np.nan
            return np.full_like(theta, r_val)
        
        # Цвета для поверхностей
        color_erg_pl = (0.0, 1.0, 0.0, 0.4)
        color_erg_in = (0.53, 0.0, 0.87, 0.4)
        color_pl = (1.0, 0.0, 0.0, 0.4)
        color_in = (1.0, 0.0, 1.0, 0.4)

        # Отрисовываем эргосферы только если есть вращение
        if not L == 0:
            self._draw_surface(r_erg_pl, color_erg_pl, None, 30, 'Внешняя эргосфера')
            self._draw_surface(r_erg_in, color_erg_in, None, 10, 'Внутренняя эргосфера')

        self._draw_surface(r_pl, color_pl, None, 30, 'Внешний горизонт событий')
        self._draw_surface(r_in, color_in, None, 10, 'Внутренний горизонт событий')

        # Устанавливаем заголовок в зависимости от направления вращения
        if L > 0:
            ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах.\n'+
                             f'M={M}, L={L}, Q={Q}. Вращение в направлении увелечения координаты φ.\n'+
                             f'[вокруг оси Z по часовой]', fontsize=12)
        elif L < 0:
            ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах\n'+
                             f'M={M}, L={L}, Q={Q}. Вращение в направлении уменьшения координаты φ.\n'+
                             f'[вокруг оси Z против часовой]', fontsize=12)
        else:
            ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах\n'+
                             f'M={M}, L={L}, Q={Q}. Вращение отсутствует.', fontsize=12)
    
    def _setup_axes(self, trajectories: np.ndarray, point_counts: np.ndarray, 
                   avto_mashtab: bool, axis_scaling_factor: float):
        """Настройка осей и масштабирования"""
        # Настройки осей
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        
        # Автомасштабирование
        if avto_mashtab:
            # Собираем все точки для определения диапазона
            all_points = []
            for idx in range(len(point_counts)):
                num_points = point_counts[idx]
                if num_points > 0:
                    traj = trajectories[idx][:num_points]
                    if traj.shape[1] >= 4:  # Проверяем, что есть координаты
                        x = traj[:, 1]
                        y = traj[:, 2]
                        z = traj[:, 3]
                        all_points.extend(zip(x, y, z))
            
            if all_points:
                all_points = np.array(all_points)
                max_val = np.max(all_points)
                min_val = np.min(all_points)
                max_range = max(np.abs(max_val), np.abs(min_val)) * 1.1
                
                self.ax.set_xlim(-max_range, max_range)
                self.ax.set_ylim(-max_range, max_range)
                self.ax.set_zlim(-max_range, max_range)
        else:
            self.ax.set_xlim(-25 * axis_scaling_factor, 25 * axis_scaling_factor)
            self.ax.set_ylim(-25 * axis_scaling_factor, 25 * axis_scaling_factor)
            self.ax.set_zlim(-17 * axis_scaling_factor, 17 * axis_scaling_factor)
        
        self.ax.grid(True)
    
    def plot(self, trajectories: np.ndarray, point_counts: np.ndarray, total_ray: int,
             coordinates_type: CoordinatesType = CoordinatesType.CARTESIAN,
             type_surfaces: MetricType = MetricType.SCHWARZSCHILD,
             step_ray: int = 1, 
             axis_scaling_factor: float = 1.0, 
             avto_mashtab: bool = True, 
             black_hole: bool = True,
             grids_of_spaces: bool = False
             ):
        """
        Построение 3D графика траекторий геодезических
        
        Args:
            trajectories: массив траекторий
            point_counts: количество точек для каждой траектории
            W: ширина сетки лучей
            H: высота сетки лучей
            coordinates_type: тип координат
            type_surfaces: тип метрики для отображения поверхностей
            step_ray: шаг отрисовки лучей
            axis_scaling_factor: коэффициент масштабирования осей
            avto_mashtab: автоматическое масштабирование
            black_hole: отображать ли поверхности черной дыры
        """
        self.trajectory_count = 0
        
        # Цикл по всем траекториям с заданным шагом
        for idx in range(0, total_ray, step_ray):
            num_points = point_counts[idx]
            if num_points < 2:
                continue
            
            # Извлекаем координаты в зависимости от типа
            traj = trajectories[idx][:num_points]
            
            if coordinates_type == CoordinatesType.CARTESIAN:
                t = traj[:, 0]
                x = traj[:, 1]
                y = traj[:, 2]
                z = traj[:, 3]
            elif coordinates_type == CoordinatesType.SPHERICAL:
                t = traj[:, 0]
                r = traj[:, 1]  # Радиальная координата
                theta = traj[:, 2]  # Угол theta (полярный угол)
                phi = traj[:, 3]  # Угол phi (азимутальный угол)

                # Фильтрация точек за горизонтом
                if black_hole:
                    valid = r > 1.001 * self.default_params['r_s']
                    r = r[valid]
                    theta = theta[valid]
                    phi = phi[valid]
                
                if len(r) < 2:
                    continue
                    
                # Преобразование координат
                x, y, z = self._spherical_to_cartesian(r, theta, phi)
            elif coordinates_type == CoordinatesType.CYLINDRICAL:
                t = traj[:, 0]
                r = traj[:, 1]    # Радиальная координата
                phi = traj[:, 2]  # Угол phi
                z_val = traj[:, 3]  # координата z

                x, y, z = self._cylindrical_to_cartesian(r, phi, z_val)
            elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                nu = traj[:, 1]
                theta = traj[:, 2]
                phi = traj[:, 3]

                x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
            elif coordinates_type == CoordinatesType.HYPERSPHERIC:
                t = traj[:, 0]
                chi = traj[:, 1]
                theta = traj[:, 2]
                phi = traj[:, 3]

                # Преобразование координат
                x, y, z = self._hyperspheric_to_cartesian_orthogonal(t, chi, theta, phi)
            else:
                # По умолчанию используем декартовы координаты
                x = traj[:, 1]
                y = traj[:, 2]
                z = traj[:, 3]

            # Вычисление цвета на основе индекса траектории
            color_value = idx / (total_ray)
            color = self.color_map(color_value)
            
            # Построение траектории
            self.ax.plot(x, y, z, 
                        linewidth=0.7, 
                        alpha=0.8,
                        color=color)
            
            self.trajectory_count += 1


        # Отрисовка справочной сетки для специфичных координат
        if grids_of_spaces:
            if coordinates_type == CoordinatesType.HYPERSPHERIC:
                self._draw_reference_grid(np.pi/2, self._hyperspheric_to_cartesian_orthogonal)
            elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                self._draw_reference_special_cylindrical_grid(self.default_params['scale_factor'])

        print(f"Построено траекторий: {self.trajectory_count}")

        # Отрисовка поверхностей черной дыры
        if black_hole:
            if type_surfaces == MetricType.SCHWARZSCHILD:
                self._draw_schwarzschild_surfaces()
            elif type_surfaces == MetricType.ELLIS_BRONNIKIVA:
                self._draw_ellis_bronnikova_surfaces()
            elif type_surfaces == MetricType.KERR_NEWMAN:
                self._draw_kerr_newman_surfaces()

        # Настройка осей
        self._setup_axes(trajectories, point_counts, avto_mashtab, axis_scaling_factor)

        plt.tight_layout()
        plt.show()


    def set_camera_position(self, elevation: float = None, azimuth: float = None, distance: float = None):
        """
        Установка положения камеры
        
        Args:
            elevation: угол возвышения (в градусах)
            azimuth: азимутальный угол (в градусах)
            distance: расстояние камеры
        """
        if elevation is not None:
            self.ax.elev = elevation
        if azimuth is not None:
            self.ax.azim = azimuth
        if distance is not None:
            self.ax.dist = distance
    
    def get_camera_position(self) -> Tuple[float, float, float]:
        """
        Получение текущего положения камеры
        
        Returns:
            Кортеж (elevation, azimuth, distance)
        """
        return self.ax.elev, self.ax.azim, self.ax.dist
    
    def set_camera_target(self, target: Tuple[float, float, float]):
        """
        Установка точки, на которую направлена камера
        
        Args:
            target: кортеж (x, y, z) - точка, на которую должна быть направлена камера
        """
        # Получаем текущее положение камеры
        elev, azim, dist = self.get_camera_position()
        
        # Вычисляем направление камеры к целевой точке
        # Для этого нужно преобразовать углы в вектор направления
        # и установить соответствующее положение камеры
        
        # Преобразуем углы в радианы
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        
        # Вычисляем вектор направления камеры
        direction = np.array([
            np.cos(elev_rad) * np.sin(azim_rad),
            np.cos(elev_rad) * np.cos(azim_rad),
            np.sin(elev_rad)
        ])
        
        # Вычисляем положение камеры
        camera_pos = np.array(target) - direction * dist
        
        # Устанавливаем новые ограничения осей для центрирования на целевой точке
        x_center, y_center, z_center = target
        
        # Получаем текущие диапазоны осей
        x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        z_range = self.ax.get_zlim()[1] - self.ax.get_zlim()[0]
        
        # Устанавливаем новые ограничения, центрированные на целевой точке
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        self.ax.set_zlim(z_center - z_range/2, z_center + z_range/2)
    
    def set_view_angles(self, elev: float = 30, azim: float = -60, roll: float = 0):
        """
        Установка углов обзора камеры
        
        Args:
            elev: угол возвышения (в градусах)
            azim: азимутальный угол (в градусах)
            roll: угол крена (в градусах)
        """
        self.ax.view_init(elev=elev, azim=azim)
        # К сожалению, matplotlib не поддерживает прямое управление креном камеры
    
    def save_camera_position(self, filename: str):
        """
        Сохранение положения камеры в файл
        
        Args:
            filename: имя файла для сохранения
        """
        elev, azim, dist = self.get_camera_position()
        camera_data = {
            'elevation': elev,
            'azimuth': azim,
            'distance': dist,
            'xlim': self.ax.get_xlim(),
            'ylim': self.ax.get_ylim(),
            'zlim': self.ax.get_zlim()
        }
        np.save(filename, camera_data)
        print(f"Положение камеры сохранено в файл: {filename}")
    
    def load_camera_position(self, filename: str):
        """
        Загрузка положения камеры из файла
        
        Args:
            filename: имя файла для загрузки
        """
        try:
            camera_data = np.load(filename, allow_pickle=True).item()
            self.set_camera_position(
                elevation=camera_data['elevation'],
                azimuth=camera_data['azimuth'],
                distance=camera_data['distance']
            )
            self.ax.set_xlim(camera_data['xlim'])
            self.ax.set_ylim(camera_data['ylim'])
            self.ax.set_zlim(camera_data['zlim'])
            print(f"Положение камеры загружено из файла: {filename}")
        except Exception as e:
            print(f"Ошибка при загрузке положения камеры: {e}")
    
    def create_rotation_animation(self, trajectories: np.ndarray, point_counts: np.ndarray, 
                                 total_ray: int, output_filename: str, fps: int = 30,
                                 rotation_speed: float = 1.0, total_rotation: float = 360.0,
                                 **kwargs):
        """
        Создание анимации с вращающейся камерой
        
        Args:
            trajectories: массив траекторий
            point_counts: количество точек для каждой траектории
            W: ширина сетки лучей
            H: высота сетки лучей
            output_filename: имя выходного файла
            fps: кадров в секунду
            rotation_speed: скорость вращения (градусов в секунду)
            total_rotation: общий угол вращения (градусов)
            **kwargs: дополнительные параметры для plot
        """
        # Вычисляем количество кадров для анимации
        duration = total_rotation / rotation_speed  # продолжительность в секундах
        total_frames = int(duration * fps)
        
        # Создаем новую фигуру для анимации
        anim_fig = plt.figure(figsize=self.fig.get_size_inches())
        anim_ax = anim_fig.add_subplot(111, projection='3d')
        
        # Строим статический график
        self._plot_static_elements(anim_ax, trajectories, point_counts, total_ray, **kwargs)
        
        # Получаем начальное положение камеры
        initial_elev, initial_azim, initial_dist = self.get_camera_position()
        
        # Функция инициализации анимации
        def init():
            return []
        
        # Функция обновления кадра
        def update(frame):
            # Вычисляем новый азимутальный угол
            new_azim = initial_azim + frame * (total_rotation / total_frames)
            anim_ax.view_init(elev=initial_elev, azim=new_azim)
            anim_ax.set_title(f'Вращение камеры: {new_azim:.1f}°')
            return []
        
        # Создаем анимацию
        ani = FuncAnimation(anim_fig, update, frames=total_frames,
                            init_func=init, blit=True, repeat=True)
        
        # Сохраняем анимацию
        try:
            writer = FFMpegWriter(fps=fps, bitrate=5000)
            ani.save(output_filename, writer=writer, dpi=100)
            print(f"Анимация вращения сохранена в файл: {output_filename}")
        except Exception as e:
            print(f"Ошибка при сохранении анимации: {e}")
        
        plt.close(anim_fig)
    
    def _plot_static_elements(self, ax, trajectories, point_counts, total_ray, **kwargs):
        """
        Вспомогательная функция для построения статических элементов
        
        Args:
            ax: ось для построения
            trajectories: массив траекторий
            point_counts: количество точек для каждой траектории
            W: ширина сетки лучей
            H: высота сетки лучей
            **kwargs: дополнительные параметры
        """
        # Копируем параметры из kwargs или используем значения по умолчанию
        coordinates_type = kwargs.get('coordinates_type', CoordinatesType.CARTESIAN)
        type_surfaces = kwargs.get('type_surfaces', MetricType.SCHWARZSCHILD)
        step_ray = kwargs.get('step_ray', 1)
        black_hole = kwargs.get('black_hole', True)
        grids_of_spaces = kwargs.get('grids_of_spaces', False)
        
        # Отрисовываем траектории
        for idx in range(0, total_ray, step_ray):
            num_points = point_counts[idx]
            if num_points < 2:
                continue
            
            # Извлекаем координаты в зависимости от типа
            traj = trajectories[idx][:num_points]
            
            if coordinates_type == CoordinatesType.CARTESIAN:
                x = traj[:, 1]
                y = traj[:, 2]
                z = traj[:, 3]
            elif coordinates_type == CoordinatesType.SPHERICAL:
                r = traj[:, 1]
                theta = traj[:, 2]
                phi = traj[:, 3]
                
                # Фильтрация точек за горизонтом
                if black_hole:
                    valid = r > 1.001 * self.default_params['r_s']
                    r = r[valid]
                    theta = theta[valid]
                    phi = phi[valid]
                
                if len(r) < 2:
                    continue
                    
                # Преобразование координат
                x, y, z = self._spherical_to_cartesian(r, theta, phi)
            # ... (аналогично для других типов координат)
            
            # Вычисление цвета на основе индекса траектории
            color_value = idx / (total_ray)
            color = self.color_map(color_value)
            
            # Построение траектории
            ax.plot(x, y, z, linewidth=0.7, alpha=0.8, color=color)
        
        # Отрисовываем статические элементы (поверхности черной дыры)
        if black_hole:
            if type_surfaces == MetricType.SCHWARZSCHILD:
                self._draw_schwarzschild_surfaces(ax)
            elif type_surfaces == MetricType.ELLIS_BRONNIKIVA:
                self._draw_ellis_bronnikova_surfaces(ax)
            elif type_surfaces == MetricType.KERR_NEWMAN:
                self._draw_kerr_newman_surfaces(ax)
        
        # Отрисовываем справочные сетки
        if grids_of_spaces:
            if coordinates_type == CoordinatesType.HYPERSPHERIC:
                self._draw_reference_grid(np.pi/2, self._hyperspheric_to_cartesian_orthogonal, ax)
            elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                self._draw_reference_special_cylindrical_grid(self.default_params['scale_factor'], ax)
        
        # Настройки осей
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
    
    def save(self, filename: str, dpi: int = 300):
        """Сохранение графика в файл"""
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"График сохранен в файл: {filename}")
    
    def clear(self):
        """Очистка графика"""
        self.ax.clear()
        self.trajectory_count = 0








import tempfile
import shutil
import subprocess
import os
from typing import Tuple, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

class TrajectoryAnimator(TrajectoryVisualizer):
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Инициализация аниматора траекторий
        
        Args:
            figsize: размер фигуры matplotlib
        """
        super().__init__(figsize)
        self.animation = None

    # Переопределяем методы отрисовки поверхностей для работы с переданной осью
    def _draw_schwarzschild_surfaces(self, ax):
        """Отрисовка поверхностей для метрики Шварцшильда на указанной оси"""
        # Горизонт событий
        self._draw_surface(self.default_params['r_s'], 'black', 0.7, 50, 'Горизонт событий', ax)
        
        # Фотонная сфера
        phi = np.linspace(0, 2*np.pi, 100)
        r_photon = 1.5 * self.default_params['r_s']
        
        # Фотонная сфера (круг в экваториальной плоскости)
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
        
        ax.set_title('Траектории света в метрике Шварцшильда\n(3D декартовы координаты)', fontsize=14)
    
    def _draw_ellis_bronnikova_surfaces(self, ax):
        """Отрисовка поверхностей для метрики Эллиса-Бронникова на указанной оси"""
        self._draw_surface(self.default_params['r_0'], 'black', 0.7, 50, 'Горловина червоточины', ax)
        ax.set_title('Траектории света в метрике Эллиса-Бронникова\n(3D декартовы координаты)', fontsize=14)
    
    def _draw_kerr_newman_surfaces(self, ax):
        """Отрисовка поверхностей для метрики Керра-Ньюмена на указанной оси"""
        M = self.default_params['M']
        Q = self.default_params['Q']
        L = self.default_params['L']
        
        # Радиус эргосфер
        def r_erg_pl(theta, phi):
            param = M**2 - Q**2 - (L/M)**2 * np.cos(theta)**2
            return np.where(param >= 0, M + np.sqrt(param), np.nan)
        
        def r_erg_in(theta, phi):
            param = M**2 - Q**2 - (L/M)**2 * np.cos(theta)**2
            return np.where(param >= 0, M - np.sqrt(param), np.nan)
        
        # Радиусы горизонтов
        def r_pl(theta, phi):
            param = M**2 - Q**2 - (L/M)**2
            r_val = M + np.sqrt(param) if param >= 0 else np.nan
            return np.full_like(theta, r_val)
        
        def r_in(theta, phi):
            param = M**2 - Q**2 - (L/M)**2
            r_val = M - np.sqrt(param) if param >= 0 else np.nan
            return np.full_like(theta, r_val)
        
        # Цвета для поверхностей
        color_erg_pl = (0.0, 1.0, 0.0, 0.4)
        color_erg_in = (0.53, 0.0, 0.87, 0.4)
        color_pl = (1.0, 0.0, 0.0, 0.4)
        color_in = (1.0, 0.0, 1.0, 0.4)

        # Отрисовываем эргосферы только если есть вращение
        if not L == 0:
            self._draw_surface(r_erg_pl, color_erg_pl, None, 30, 'Внешняя эргосфера', ax)
            self._draw_surface(r_erg_in, color_erg_in, None, 10, 'Внутренняя эргосфера', ax)

        self._draw_surface(r_pl, color_pl, None, 30, 'Внешний горизонт событий', ax)
        self._draw_surface(r_in, color_in, None, 10, 'Внутренний горизонт событий', ax)

        # Устанавливаем заголовок в зависимости от направления вращения
        if L > 0:
            ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах.\n'+
                         f'M={M}, L={L}, Q={Q}. Вращение в направлении увелечения координаты φ.\n'+
                         f'[вокруг оси Z по часовой]', fontsize=12)
        elif L < 0:
            ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах\n'+
                         f'M={M}, L={L}, Q={Q}. Вращение в направлении уменьшения координаты φ.\n'+
                         f'[вокруг оси Z против часовой]', fontsize=12)
        else:
            ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах\n'+
                         f'M={M}, L={L}, Q={Q}. Вращение отсутствует.', fontsize=12)
    
    def _draw_surface(self, r, color: str, alpha: float, num: int, name: str, ax):
        """
        Построение поверхности на указанной оси
        
        Args:
            r: радиус или функция, возвращающая радиус
            color: цвет поверхности
            alpha: прозрачность
            num: количество точек для построения
            name: название поверхности для легенды
            ax: ось для отрисовки
        """
        phi = np.linspace(0, 2 * np.pi, num)
        theta = np.linspace(0, np.pi, num)
        phi, theta = np.meshgrid(phi, theta)

        # Проверяем, является ли параметр функцией или переменной
        if callable(r):
            R = r(theta, phi)
        else:
            R = r

        X = R * np.sin(theta) * np.cos(phi)
        Y = R * np.sin(theta) * np.sin(phi)
        Z = R * np.cos(theta)

        surf = ax.plot_surface(X, Y, Z, color=color, alpha=alpha, label=name)
        return surf
    
    def _draw_reference_grid(self, chi: float, conversion_func: Callable, ax):
        """Отрисовка справочной сетки для гиперсферических координат на указанной оси"""
        theta_mass = np.linspace(1e-10, np.pi, 10)
        for theta in theta_mass:
            phi = np.linspace(1e-10, 2*np.pi, 100)
            x, y, z = conversion_func(1, chi, theta, phi)
            ax.plot(x, y, z, linewidth=0.7, color=(1.0, 0.0, 0.0, 0.5))

        phi_mass = np.linspace(1e-10, 2*np.pi, 10)
        for phi in phi_mass:
            theta = np.linspace(1e-10, 2*np.pi, 100)
            x, y, z = conversion_func(1, chi, theta, phi)
            ax.plot(x, y, z, linewidth=0.7, color=(0.0, 1.0, 0.0, 0.5))
    
    def _draw_reference_special_cylindrical_grid(self, R, ax):
        """Отрисовка справочной сетки для специальных цилиндрических координат на указанной оси"""
        theta = np.pi/2

        phi_mass = np.linspace(1e-10, 2*np.pi, 10)
        for phi in phi_mass:
            nu = np.linspace(-20, 20, 5)
            x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
            ax.plot(x, y, z, linewidth=0.7, color=(1.0, 0.0, 0.0, 0.7))

        nu_mass = np.linspace(-20, 20, 10)
        for nu in nu_mass:
            phi = np.linspace(1e-10, 2*np.pi, 50)
            x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
            ax.plot(x, y, z, linewidth=0.7, color=(1.0, 0.0, 0.0, 0.7))

    def _save_frames_and_build_video(self, anim_fig, update_func, total_frames, output_filename, 
                                   fps=30, dpi=100, progress_callback=None):
        """
        Вспомогательный метод для сохранения кадров и сборки видео через FFmpeg
        
        Args:
            anim_fig: фигура matplotlib
            update_func: функция обновления кадра
            total_frames: общее количество кадров
            output_filename: имя выходного файла
            fps: кадров в секунду
            dpi: разрешение
            progress_callback: функция для отслеживания прогресса
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Сохраняем все кадры во временную директорию
            for frame in range(total_frames):
                if progress_callback:
                    progress_callback(frame, total_frames)
                
                # Обновляем кадр
                update_func(frame)
                
                # Сохраняем кадр
                frame_path = os.path.join(temp_dir, f"frame_{frame:06d}.png")
                anim_fig.savefig(frame_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            
            # Собираем видео из кадров с помощью FFmpeg
            file_ext = os.path.splitext(output_filename)[1].lower()
            
            if file_ext == '.gif':
                # Для GIF используем палитру для лучшего качества
                palette_path = os.path.join(temp_dir, "palette.png")
                palette_cmd = [
                    'ffmpeg', '-y',
                    '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                    '-vf', 'palettegen',
                    palette_path
                ]
                
                try:
                    subprocess.run(palette_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(f"Ошибка при создании палитры: {e.stderr.decode()}")
                    return
                
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-r', str(fps),
                    '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                    '-i', palette_path,
                    '-filter_complex', 'paletteuse',
                    '-r', str(fps),
                    output_filename
                ]
            else:
                # Для видео используем стандартные настройки
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-r', str(fps),
                    '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',
                    '-preset', 'medium',
                    output_filename
                ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Анимация успешно сохранена в {output_filename}")
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при вызове FFmpeg: {e.stderr.decode()}")
            except FileNotFoundError:
                print("FFmpeg не найден. Убедитесь, что FFmpeg установлен и добавлен в PATH")
        
        finally:
            # Удаляем временную директорию
            shutil.rmtree(temp_dir, ignore_errors=True)

    def create_animation(self, trajectories: np.ndarray, point_counts: np.ndarray, 
                        total_ray: int, output_filename: str, fps: int = 30,
                        coordinates_type: CoordinatesType = CoordinatesType.SPHERICAL,
                        type_surfaces: MetricType = MetricType.SCHWARZSCHILD,
                        step_ray: int = 1, 
                        axis_scaling_factor: float = 1.0, 
                        avto_mashtab: bool = True, 
                        black_hole: bool = True,
                        grids_of_spaces: bool = False,
                        dpi: int = 100,
                        progress_callback: Optional[Callable] = None):
        """
        Создание анимации из траекторий
        
        Args:
            trajectories: массив траекторий формы (N, M, 8) где N - количество лучей, M - количество точек
            point_counts: массив с количеством точек для каждого луча
            total_ray: общее количество лучей
            output_filename: имя выходного файла
            fps: кадров в секунду
            coordinates_type: тип координат
            type_surfaces: тип метрики для отображения поверхностей
            step_ray: шаг отрисовки лучей
            axis_scaling_factor: коэффициент масштабирования осей
            avto_mashtab: автоматическое масштабирование
            black_hole: отображать ли поверхности черной дыры
            grids_of_spaces: отображать ли справочные сетки
            dpi: разрешение видео
            progress_callback: функция для отслеживания прогресса
        """
        # Создаем новую фигуру для анимации
        anim_fig = plt.figure(figsize=self.fig.get_size_inches())
        anim_ax = anim_fig.add_subplot(111, projection='3d')
        
        # Определяем общее количество кадров (минимальное количество точек среди всех траекторий)
        total_frames = np.max(point_counts)
        
        # Предварительно вычисляем диапазон координат для всего набора данных
        all_points = []
        for idx in range(0, total_ray, step_ray):
            num_points = point_counts[idx]
            if num_points > 0:
                traj = trajectories[idx][:num_points]
                if traj.shape[1] >= 4:  # Проверяем, что есть координаты
                    if coordinates_type == CoordinatesType.SPHERICAL:
                        r = traj[:, 1]
                        theta = traj[:, 2]
                        phi = traj[:, 3]
                        x, y, z = self._spherical_to_cartesian(r, theta, phi)
                    elif coordinates_type == CoordinatesType.CYLINDRICAL:
                        r = traj[:, 1]
                        phi_vals = traj[:, 2]
                        z_val = traj[:, 3]
                        x, y, z = self._cylindrical_to_cartesian(r, phi_vals, z_val)
                    elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                        nu = traj[:, 1]
                        theta = traj[:, 2]
                        phi = traj[:, 3]
                        x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
                    elif coordinates_type == CoordinatesType.HYPERSPHERIC:
                        t = traj[:, 0]
                        chi = traj[:, 1]
                        theta = traj[:, 2]
                        phi = traj[:, 3]
                        x, y, z = self._hyperspheric_to_cartesian_orthogonal(t, chi, theta, phi)
                    else:
                        x = traj[:, 1]
                        y = traj[:, 2]
                        z = traj[:, 3]
                    
                    all_points.extend(zip(x, y, z))
        
        # Устанавливаем диапазон осей
        if all_points:
            all_points = np.array(all_points)
            max_val = np.max(all_points)
            min_val = np.min(all_points)
            max_range = max(np.abs(max_val), np.abs(min_val)) * 1.1
            
            anim_ax.set_xlim(-max_range, max_range)
            anim_ax.set_ylim(-max_range, max_range)
            anim_ax.set_zlim(-max_range, max_range)
        
        # Настройки осей
        anim_ax.set_xlabel('X')
        anim_ax.set_ylabel('Y')
        anim_ax.set_zlabel('Z')
        anim_ax.grid(True)
        
        # Отрисовываем статические элементы (поверхности черной дыры)
        if black_hole:
            if type_surfaces == MetricType.SCHWARZSCHILD:
                self._draw_schwarzschild_surfaces(ax=anim_ax)
            elif type_surfaces == MetricType.ELLIS_BRONNIKIVA:
                self._draw_ellis_bronnikova_surfaces(ax=anim_ax)
            elif type_surfaces == MetricType.KERR_NEWMAN:
                self._draw_kerr_newman_surfaces(ax=anim_ax)
        
        # Отрисовываем справочные сетки
        if grids_of_spaces:
            if coordinates_type == CoordinatesType.HYPERSPHERIC:
                self._draw_reference_grid(np.pi/2, self._hyperspheric_to_cartesian_orthogonal, ax=anim_ax)
            elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                self._draw_reference_special_cylindrical_grid(self.default_params['scale_factor'], ax=anim_ax)
        
        # Создаем пустые линии для траекторий
        lines = []
        for idx in range(0, total_ray, step_ray):
            color_value = idx / total_ray
            color = self.color_map(color_value)
            line, = anim_ax.plot([], [], [], linewidth=0.7, alpha=0.8, color=color)
            lines.append(line)
        
        # Функция обновления кадра
        def update_frame(frame):
            line_idx = 0
            for idx in range(0, total_ray, step_ray):
                num_points = point_counts[idx]
                if num_points <= frame:
                    lines[line_idx].set_data([], [])
                    lines[line_idx].set_3d_properties([])
                    line_idx += 1
                    continue
                
                # Извлекаем координаты в зависимости от типа
                traj = trajectories[idx][:frame+1]  # Берем точки до текущего кадра
                
                if coordinates_type == CoordinatesType.CARTESIAN:
                    x = traj[:, 1]
                    y = traj[:, 2]
                    z = traj[:, 3]
                elif coordinates_type == CoordinatesType.SPHERICAL:
                    r = traj[:, 1]
                    theta = traj[:, 2]
                    phi = traj[:, 3]
                    
                    # Фильтрация точек за горизонтом
                    if black_hole:
                        valid = r > 1.001 * self.default_params['r_s']
                        r = r[valid]
                        theta = theta[valid]
                        phi = phi[valid]
                    
                    if len(r) < 2:
                        lines[line_idx].set_data([], [])
                        lines[line_idx].set_3d_properties([])
                        line_idx += 1
                        continue
                        
                    # Преобразование координат
                    x, y, z = self._spherical_to_cartesian(r, theta, phi)
                elif coordinates_type == CoordinatesType.CYLINDRICAL:
                    r = traj[:, 1]
                    phi_vals = traj[:, 2]
                    z_val = traj[:, 3]
                    
                    x, y, z = self._cylindrical_to_cartesian(r, phi_vals, z_val)
                elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                    nu = traj[:, 1]
                    theta = traj[:, 2]
                    phi = traj[:, 3]
                    
                    x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
                elif coordinates_type == CoordinatesType.HYPERSPHERIC:
                    t = traj[:, 0]
                    chi = traj[:, 1]
                    theta = traj[:, 2]
                    phi = traj[:, 3]
                    
                    # Преобразование координат
                    x, y, z = self._hyperspheric_to_cartesian_orthogonal(t, chi, theta, phi)
                else:
                    # По умолчанию используем декартовы координаты
                    x = traj[:, 1]
                    y = traj[:, 2]
                    z = traj[:, 3]
                
                # Обновляем данные линии
                lines[line_idx].set_data(x, y)
                lines[line_idx].set_3d_properties(z)
                line_idx += 1
            
            # Обновляем заголовок с номером кадра
            anim_ax.set_title(f'Кадр {frame+1}/{total_frames}')
            
            return lines
        
        # Используем новый метод для сохранения анимации
        self._save_frames_and_build_video(
            anim_fig, update_frame, total_frames, output_filename, 
            fps, dpi, progress_callback
        )
        
        plt.close(anim_fig)
        return None

    def create_animation_with_rotation(self, trajectories: np.ndarray, point_counts: np.ndarray, 
                                     total_ray: int, output_filename: str, fps: int = 30,
                                     coordinates_type: CoordinatesType = CoordinatesType.SPHERICAL,
                                     type_surfaces: MetricType = MetricType.SCHWARZSCHILD,
                                     step_ray: int = 1, 
                                     axis_scaling_factor: float = 1.0, 
                                     avto_mashtab: bool = True, 
                                     black_hole: bool = True,
                                     grids_of_spaces: bool = False,
                                     dpi: int = 100,
                                     progress_callback: Optional[Callable] = None,
                                     rotation_speed: float = 0.5,
                                     rotation_axis: str = 'z',
                                     initial_elev: float = 30,
                                     initial_azim: float = -60):
        """
        Создание анимации из траекторий с вращением камеры
        
        Args:
            trajectories: массив траекторий формы (N, M, 8) где N - количество лучей, M - количество точек
            point_counts: массив с количеством точек для каждого луча
            total_ray: общее количество лучей
            output_filename: имя выходного файла
            fps: кадров в секунду
            coordinates_type: тип координат
            type_surfaces: тип метрики для отображения поверхностей
            step_ray: шаг отрисовки лучей
            axis_scaling_factor: коэффициент масштабирования осей
            avto_mashtab: автоматическое масштабирование
            black_hole: отображать ли поверхности черной дыры
            grids_of_spaces: отображать ли справочные сетки
            dpi: разрешение видео
            progress_callback: функции для отслеживания прогресса
            rotation_speed: скорость вращения камеры (градусов за кадр)
            rotation_axis: ось вращения ('x', 'y', 'z' или 'all')
            initial_elev: начальный угол возвышения камеры
            initial_azim: начальный азимутальный угол камеры
        """
        # Создаем новую фигуру для анимации
        anim_fig = plt.figure(figsize=self.fig.get_size_inches())
        anim_ax = anim_fig.add_subplot(111, projection='3d')
        
        # Устанавливаем начальное положение камеры
        anim_ax.view_init(elev=initial_elev, azim=initial_azim)
        
        # Определяем общее количество кадров (минимальное количество точек среди всех траекторий)
        total_frames = np.max(point_counts)
        
        # Предварительно вычисляем диапазон координат для всего набора данных
        all_points = []
        for idx in range(0, total_ray, step_ray):
            num_points = point_counts[idx]
            if num_points > 0:
                traj = trajectories[idx][:num_points]
                if traj.shape[1] >= 4:  # Проверяем, что есть координаты
                    if coordinates_type == CoordinatesType.SPHERICAL:
                        r = traj[:, 1]
                        theta = traj[:, 2]
                        phi = traj[:, 3]
                        x, y, z = self._spherical_to_cartesian(r, theta, phi)
                    elif coordinates_type == CoordinatesType.CYLINDRICAL:
                        r = traj[:, 1]
                        phi_vals = traj[:, 2]
                        z_val = traj[:, 3]
                        x, y, z = self._cylindrical_to_cartesian(r, phi_vals, z_val)
                    elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                        nu = traj[:, 1]
                        theta = traj[:, 2]
                        phi = traj[:, 3]
                        x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
                    elif coordinates_type == CoordinatesType.HYPERSPHERIC:
                        t = traj[:, 0]
                        chi = traj[:, 1]
                        theta = traj[:, 2]
                        phi = traj[:, 3]
                        x, y, z = self._hyperspheric_to_cartesian_orthogonal(t, chi, theta, phi)
                    else:
                        x = traj[:, 1]
                        y = traj[:, 2]
                        z = traj[:, 3]
                    
                    all_points.extend(zip(x, y, z))
        
        # Устанавливаем диапазон осей
        if all_points:
            all_points = np.array(all_points)
            max_val = np.max(all_points)
            min_val = np.min(all_points)
            max_range = max(np.abs(max_val), np.abs(min_val)) * 1.1
            
            anim_ax.set_xlim(-max_range, max_range)
            anim_ax.set_ylim(-max_range, max_range)
            anim_ax.set_zlim(-max_range, max_range)
        
        # Настройки осей
        anim_ax.set_xlabel('X')
        anim_ax.set_ylabel('Y')
        anim_ax.set_zlabel('Z')
        anim_ax.grid(True)
        
        # Отрисовываем статические элементы (поверхности черной дыры)
        if black_hole:
            if type_surfaces == MetricType.SCHWARZSCHILD:
                self._draw_schwarzschild_surfaces(ax=anim_ax)
            elif type_surfaces == MetricType.ELLIS_BRONNIKIVA:
                self._draw_ellis_bronnikova_surfaces(ax=anim_ax)
            elif type_surfaces == MetricType.KERR_NEWMAN:
                self._draw_kerr_newman_surfaces(ax=anim_ax)
        
        # Отрисовываем справочные сетки
        if grids_of_spaces:
            if coordinates_type == CoordinatesType.HYPERSPHERIC:
                self._draw_reference_grid(np.pi/2, self._hyperspheric_to_cartesian_orthogonal, ax=anim_ax)
            elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                self._draw_reference_special_cylindrical_grid(self.default_params['scale_factor'], ax=anim_ax)
        
        # Создаем пустые линии для траекторий
        lines = []
        for idx in range(0, total_ray, step_ray):
            color_value = idx / total_ray
            color = self.color_map(color_value)
            line, = anim_ax.plot([], [], [], linewidth=0.7, alpha=0.8, color=color)
            lines.append(line)
        
        # Функция обновления кадра
        def update_frame(frame):
            line_idx = 0
            for idx in range(0, total_ray, step_ray):
                num_points = point_counts[idx]
                if num_points <= frame:
                    lines[line_idx].set_data([], [])
                    lines[line_idx].set_3d_properties([])
                    line_idx += 1
                    continue
                
                # Извлекаем координаты в зависимости от типа
                traj = trajectories[idx][:frame+1]  # Берем точки до текущего кадра
                
                if coordinates_type == CoordinatesType.CARTESIAN:
                    x = traj[:, 1]
                    y = traj[:, 2]
                    z = traj[:, 3]
                elif coordinates_type == CoordinatesType.SPHERICAL:
                    r = traj[:, 1]
                    theta = traj[:, 2]
                    phi = traj[:, 3]
                    
                    # Фильтрация точек за горизонтом
                    if black_hole:
                        valid = r > 1.001 * self.default_params['r_s']
                        r = r[valid]
                        theta = theta[valid]
                        phi = phi[valid]
                    
                    if len(r) < 2:
                        lines[line_idx].set_data([], [])
                        lines[line_idx].set_3d_properties([])
                        line_idx += 1
                        continue
                        
                    # Преобразование координат
                    x, y, z = self._spherical_to_cartesian(r, theta, phi)
                elif coordinates_type == CoordinatesType.CYLINDRICAL:
                    r = traj[:, 1]
                    phi_vals = traj[:, 2]
                    z_val = traj[:, 3]
                    
                    x, y, z = self._cylindrical_to_cartesian(r, phi_vals, z_val)
                elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                    nu = traj[:, 1]
                    theta = traj[:, 2]
                    phi = traj[:, 3]
                    
                    x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
                elif coordinates_type == CoordinatesType.HYPERSPHERIC:
                    t = traj[:, 0]
                    chi = traj[:, 1]
                    theta = traj[:, 2]
                    phi = traj[:, 3]
                    
                    # Преобразование координат
                    x, y, z = self._hyperspheric_to_cartesian_orthogonal(t, chi, theta, phi)
                else:
                    # По умолчанию используем декартовы координаты
                    x = traj[:, 1]
                    y = traj[:, 2]
                    z = traj[:, 3]
                
                # Обновляем данные линии
                lines[line_idx].set_data(x, y)
                lines[line_idx].set_3d_properties(z)
                line_idx += 1
            
            # Вращаем камеру
            if rotation_axis == 'x':
                # Вращение вокруг оси X (изменение угла возвышения)
                new_elev = initial_elev + frame * rotation_speed
                anim_ax.view_init(elev=new_elev, azim=initial_azim)
            elif rotation_axis == 'y':
                # Вращение вокруг оси Y (изменение азимутального угла)
                new_azim = initial_azim + frame * rotation_speed
                anim_ax.view_init(elev=initial_elev, azim=new_azim)
            elif rotation_axis == 'z':
                # Вращение вокруг оси Z (комбинация изменения углов)
                new_elev = initial_elev + frame * rotation_speed * 0.5
                new_azim = initial_azim + frame * rotation_speed
                anim_ax.view_init(elev=new_elev, azim=new_azim)
            elif rotation_axis == 'all':
                # Сложное вращение вокруг всех осей
                new_elev = initial_elev + frame * rotation_speed * 0.7
                new_azim = initial_azim + frame * rotation_speed * 1.2
                anim_ax.view_init(elev=new_elev, azim=new_azim)
            
            # Обновляем заголовок с номером кадра
            anim_ax.set_title(f'Кадр {frame+1}/{total_frames}')
            
            return lines
        
        # Используем новый метод для сохранения анимации
        self._save_frames_and_build_video(
            anim_fig, update_frame, total_frames, output_filename, 
            fps, dpi, progress_callback
        )
        
        plt.close(anim_fig)
        return None

    def create_orbital_animation(self, trajectories: np.ndarray, point_counts: np.ndarray, 
                               total_ray: int, output_filename: str, fps: int = 30,
                               coordinates_type: CoordinatesType = CoordinatesType.SPHERICAL,
                               type_surfaces: MetricType = MetricType.SCHWARZSCHILD,
                               step_ray: int = 1, 
                               axis_scaling_factor: float = 1.0, 
                               avto_mashtab: bool = True, 
                               black_hole: bool = True,
                               grids_of_spaces: bool = False,
                               dpi: int = 100,
                               progress_callback: Optional[Callable] = None,
                               orbit_radius: float = 10.0,
                               orbit_speed: float = 0.5):
        """
        Создание анимации с орбитальным движением камеры вокруг сцены
        
        Args:
            trajectories: массив траекторий формы (N, M, 8) где N - количество лучей, M - количество точек
            point_counts: массив с количеством точек для каждого луча
            total_ray: общее количество лучей
            output_filename: имя выходного файла
            fps: кадров в секунду
            coordinates_type: тип координат
            type_surfaces: тип метрики для отображения поверхностей
            step_ray: шаг отрисовки лучей
            axis_scaling_factor: коэффициент масштабирования осей
            avto_mashtab: автоматическое масштабирование
            black_hole: отображать ли поверхности черной дыры
            grids_of_spaces: отображать ли справочные сетки
            dpi: разрешение видео
            progress_callback: функция для отслеживания прогресса
            orbit_radius: радиус орбиты камеры
            orbit_speed: скорость движения по орбите
        """
        # Создаем новую фигуру для анимации
        anim_fig = plt.figure(figsize=self.fig.get_size_inches())
        anim_ax = anim_fig.add_subplot(111, projection='3d')
        
        # Определяем общее количество кадров (минимальное количество точек среди всех траекторий)
        total_frames = np.max(point_counts)
        
        # Предварительно вычисляем диапазон координат для всего набора данных
        all_points = []
        for idx in range(0, total_ray, step_ray):
            num_points = point_counts[idx]
            if num_points > 0:
                traj = trajectories[idx][:num_points]
                if traj.shape[1] >= 4:  # Проверяем, что есть координаты
                    if coordinates_type == CoordinatesType.SPHERICAL:
                        r = traj[:, 1]
                        theta = traj[:, 2]
                        phi = traj[:, 3]
                        x, y, z = self._spherical_to_cartesian(r, theta, phi)
                    elif coordinates_type == CoordinatesType.CYLINDRICAL:
                        r = traj[:, 1]
                        phi_vals = traj[:, 2]
                        z_val = traj[:, 3]
                        x, y, z = self._cylindrical_to_cartesian(r, phi_vals, z_val)
                    elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                        nu = traj[:, 1]
                        theta = traj[:, 2]
                        phi = traj[:, 3]
                        x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
                    elif coordinates_type == CoordinatesType.HYPERSPHERIC:
                        t = traj[:, 0]
                        chi = traj[:, 1]
                        theta = traj[:, 2]
                        phi = traj[:, 3]
                        x, y, z = self._hyperspheric_to_cartesian_orthogonal(t, chi, theta, phi)
                    else:
                        x = traj[:, 1]
                        y = traj[:, 2]
                        z = traj[:, 3]
                    
                    all_points.extend(zip(x, y, z))
        
        # Устанавливаем диапазон осей
        if all_points:
            all_points = np.array(all_points)
            max_val = np.max(all_points)
            min_val = np.min(all_points)
            max_range = max(np.abs(max_val), np.abs(min_val)) * 1.1
            
            anim_ax.set_xlim(-max_range, max_range)
            anim_ax.set_ylim(-max_range, max_range)
            anim_ax.set_zlim(-max_range, max_range)
        
        # Настройки осей
        anim_ax.set_xlabel('X')
        anim_ax.set_ylabel('Y')
        anim_ax.set_zlabel('Z')
        anim_ax.grid(True)
        
        # Отрисовываем статические элементы (поверхности черной дыры)
        if black_hole:
            if type_surfaces == MetricType.SCHWARZSCHILD:
                self._draw_schwarzschild_surfaces(ax=anim_ax)
            elif type_surfaces == MetricType.ELLIS_BRONNIKIVA:
                self._draw_ellis_bronnikova_surfaces(ax=anim_ax)
            elif type_surfaces == MetricType.KERR_NEWMAN:
                self._draw_kerr_newman_surfaces(ax=anim_ax)
        
        # Отрисовываем справочные сетки
        if grids_of_spaces:
            if coordinates_type == CoordinatesType.HYPERSPHERIC:
                self._draw_reference_grid(np.pi/2, self._hyperspheric_to_cartesian_orthogonal, ax=anim_ax)
            elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                self._draw_reference_special_cylindrical_grid(self.default_params['scale_factor'], ax=anim_ax)
        
        # Создаем пустые линии для траекторий
        lines = []
        for idx in range(0, total_ray, step_ray):
            color_value = idx / total_ray
            color = self.color_map(color_value)
            line, = anim_ax.plot([], [], [], linewidth=0.7, alpha=0.8, color=color)
            lines.append(line)
        
        # Функция обновления кадра
        def update_frame(frame):
            line_idx = 0
            for idx in range(0, total_ray, step_ray):
                num_points = point_counts[idx]
                if num_points <= frame:
                    lines[line_idx].set_data([], [])
                    lines[line_idx].set_3d_properties([])
                    line_idx += 1
                    continue
                
                # Извлекаем координаты в зависимости от типа
                traj = trajectories[idx][:frame+1]  # Берем точки до текущего кадра
                
                if coordinates_type == CoordinatesType.CARTESIAN:
                    x = traj[:, 1]
                    y = traj[:, 2]
                    z = traj[:, 3]
                elif coordinates_type == CoordinatesType.SPHERICAL:
                    r = traj[:, 1]
                    theta = traj[:, 2]
                    phi = traj[:, 3]
                    
                    # Фильтрация точек за горизонтом
                    if black_hole:
                        valid = r > 1.001 * self.default_params['r_s']
                        r = r[valid]
                        theta = theta[valid]
                        phi = phi[valid]
                    
                    if len(r) < 2:
                        lines[line_idx].set_data([], [])
                        lines[line_idx].set_3d_properties([])
                        line_idx += 1
                        continue
                        
                    # Преобразование координат
                    x, y, z = self._spherical_to_cartesian(r, theta, phi)
                elif coordinates_type == CoordinatesType.CYLINDRICAL:
                    r = traj[:, 1]
                    phi_vals = traj[:, 2]
                    z_val = traj[:, 3]
                    
                    x, y, z = self._cylindrical_to_cartesian(r, phi_vals, z_val)
                elif coordinates_type == CoordinatesType.SPECIAL_CYLINDRICAL:
                    nu = traj[:, 1]
                    theta = traj[:, 2]
                    phi = traj[:, 3]
                    
                    x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
                elif coordinates_type == CoordinatesType.HYPERSPHERIC:
                    t = traj[:, 0]
                    chi = traj[:, 1]
                    theta = traj[:, 2]
                    phi = traj[:, 3]
                    
                    # Преобразование координат
                    x, y, z = self._hyperspheric_to_cartesian_orthogonal(t, chi, theta, phi)
                else:
                    # По умолчанию используем декартовы координаты
                    x = traj[:, 1]
                    y = traj[:, 2]
                    z = traj[:, 3]
                
                # Обновляем данные линии
                lines[line_idx].set_data(x, y)
                lines[line_idx].set_3d_properties(z)
                line_idx += 1
            
            # Орбитальное движение камеры
            angle = frame * orbit_speed
            elev = 30  # Фиксированный угол возвышения
            azim = angle * 180 / np.pi  # Преобразуем радианы в градусы
            
            # Вычисляем положение камеры на орбите
            camera_x = orbit_radius * np.cos(angle)
            camera_y = orbit_radius * np.sin(angle)
            camera_z = orbit_radius * np.sin(elev * np.pi / 180)
            
            # Устанавливаем положение камеры
            anim_ax.view_init(elev=elev, azim=azim)
            
            # Обновляем заголовок с номером кадра
            anim_ax.set_title(f'Кадр {frame+1}/{total_frames}\nКамера: ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f})')
            
            return lines
        
        # Используем новый метод для сохранения анимации
        self._save_frames_and_build_video(
            anim_fig, update_frame, total_frames, output_filename, 
            fps, dpi, progress_callback
        )
        
        plt.close(anim_fig)
        return None