import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from typing import Callable, Tuple
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
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z
    
    def _cylindrical_to_cartesian(self, r: np.ndarray, phi: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Преобразование цилиндрических координат в декартовы"""
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
        a = self.default_params['static_scale_factor'] + np.exp(self.default_params['hubble_parameter'] * t)
        cot = np.cos(chi/2) / np.sin(chi/2)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        x = a * cot * sin_theta * cos_phi
        y = a * cot * sin_theta * sin_phi
        z = a * cot * cos_theta

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
    
    def _draw_reference_grid(self, chi: float, conversion_func: Callable):
        """Отрисовка справочной сетки для гиперсферических координат"""
        theta_mass = np.linspace(1e-10, np.pi, 10)
        for theta in theta_mass:
            phi = np.linspace(1e-10, 2*np.pi, 100)
            x, y, z = conversion_func(1, chi, theta, phi)
            self.ax.plot(x, y, z, linewidth=0.7, color=(1.0, 0.0, 0.0, 0.5))

        phi_mass = np.linspace(1e-10, 2*np.pi, 10)
        for phi in phi_mass:
            theta = np.linspace(1e-10, 2*np.pi, 100)
            x, y, z = conversion_func(1, chi, theta, phi)
            self.ax.plot(x, y, z, linewidth=0.7, color=(0.0, 1.0, 0.0, 0.5))

    def _draw_reference_special_cylindrical_grid(self, R):
        """Отрисовка справочной сетки для специальных цилиндрических координат"""
        theta = np.pi/2

        phi_mass = np.linspace(1e-10, 2*np.pi, 10)
        for phi in phi_mass:
            nu = np.linspace(-20, 20, 5)
            x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
            self.ax.plot(x, y, z, linewidth=0.7, color=(1.0, 0.0, 0.0, 0.7))

        nu_mass = np.linspace(-20, 20, 10)
        for nu in nu_mass:
            phi = np.linspace(1e-10, 2*np.pi, 50)
            x, y, z = self._specal_cylindical(self.default_params['scale_factor'], nu, theta, phi)
            self.ax.plot(x, y, z, linewidth=0.7, color=(1.0, 0.0, 0.0, 0.7))     
    
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
    
    def _draw_schwarzschild_surfaces(self):
        """Отрисовка поверхностей для метрики Шварцшильда"""
        # Горизонт событий
        self._draw_surface(self.default_params['r_0'], 'black', 0.7, 50, 'Горизонт событий')
        
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
        self.ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7)

        # Фотонная сфера (круг вокруг оси x)
        z_ph = r_photon * np.cos(phi)
        y_ph = r_photon * np.sin(phi)
        x_ph = np.zeros_like(phi)
        self.ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7)
        
        self.ax.set_title('Траектории света в метрике Шварцшильда\n(3D декартовы координаты)', fontsize=14)
    
    def _draw_ellis_bronnikova_surfaces(self):
        """Отрисовка поверхностей для метрики Эллиса-Бронникова"""
        self._draw_surface(self.default_params['r_0'], 'black', 0.7, 50, 'Горловина червоточины')
        self.ax.set_title('Траектории света в метрике Эллиса-Бронникова\n(3D декартовы координаты)', fontsize=14)
    
    def _draw_kerr_newman_surfaces(self):
        """Отрисовка поверхностей для метрики Керра-Ньюмена"""
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
            self.ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах.\n'+
                             f'M={M}, L={L}, Q={Q}. Вращение в направлении увелечения координаты φ.\n'+
                             f'[вокруг оси Z по часовой]', fontsize=12)
        elif L < 0:
            self.ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах\n'+
                             f'M={M}, L={L}, Q={Q}. Вращение в направлении уменьшения координаты φ.\n'+
                             f'[вокруг оси Z против часовой]', fontsize=12)
        else:
            self.ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах\n'+
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
    
    def plot(self, trajectories: np.ndarray, point_counts: np.ndarray, W: int, H: int,
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
        for idx in range(0, W * H, step_ray):
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
            color_value = idx / (W * H)
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
    
    def save(self, filename: str, dpi: int = 300):
        """Сохранение графика в файл"""
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"График сохранен в файл: {filename}")
    
    def clear(self):
        """Очистка графика"""
        self.ax.clear()
        self.trajectory_count = 0