
import numpy as np
from enums import MetricType
from vector4 import Vector4

class Metric:
    """Класс для работы с различными метриками пространства-времени"""
    def __init__(self, metric_type=MetricType.SCHWARZSCHILD,
                 r_s=1.0, 
                 r_0=1.0, 
                 M=1.0, Q=1.0, L=1.0, 
                 godel_moment=1.0,
                 scale_factor=1.0,
                 k=1.0,
                 static_scale_factor=1.0,
                 hubble_parameter=1.0
                 ):
        """
        Инициализация метрики
        
        Параметры:
            metric_type: тип метрики
                
            Метрика Шварцшильда
            r_s: радиус Шварцшильда для метрики Шварцшильда
            
            Метрика Эллиса-Бронникова
            r_0: ширина горловины кротовой норы Эллиса

            Метрика Керра-Ньюмена
            M: масса объекта
            Q: электрический заряд объекта
            L: момент вращения объекта

            Метрика Гёделя
            godel_moment: момент вращения вселенной

            Метрика Фридмана-Робетсона-Уокера
            scale_factor: масштабный фактор
            k: праметр кривизны

            Метрика сферической вселенной в гиперсферических координатах
            static_scale_factor: статичный маштабный фактор
            hubble_parameter: параметр Хаббла (скорость расширения вселенной)
            
        """
        # Тип метрики
        self.metric_type = metric_type
        
        # Параметры метриеи Шварцшильда
        self.r_s = r_s  
        
        # Параметры метрики Эллиса-Бронникова
        self.r_0 = r_0

        # Параметры метрики Гёделя
        self.godel_moment = godel_moment

        # Параметры метрики Керра
        self.M = M
        self.Q = Q
        self.L = L
        self.a = L/M

        # Параметры метрики Фридмана_Робертсона_Уокера
        # (сферическая статичная/не_расширающаяся вселенная)
        self.scale_factor = scale_factor
        self.k = k

        # Параметры сферической вселенной
        self.static_scale_factor = static_scale_factor
        self.hubble_parameter = hubble_parameter
        
        self.transform_cache = {}
    
    def get_metric_tensor(self, position, dtype=np.float64):
        """
        Возвращает метрический тензор в заданной точке
        
        Параметры:
            position: 4-вектор положения (t, r, θ, φ), (t, x, y, z), (t, r, θ, z)
            dtype: тип данных тензора
            
        Возвращает:
            g: метрический тензор (4x4)
        """
        if self.metric_type == MetricType.MINKOWSKI:
            return self.minkowski(dtype)
        elif self.metric_type == MetricType.SCHWARZSCHILD:
            return self.schwarzschild(position, self.r_s, dtype)
        elif self.metric_type == MetricType.ELLIS_BRONNIKIVA:
            return self.ellis_bronnikova(position, self.r_0, dtype)
        elif self.metric_type == MetricType.KERR_NEWMAN:
            return self.kerr_newman(position, self.M, self.Q, self.a, dtype)
        elif self.metric_type == MetricType.GOEDEL:
            return self.godel(position, self.godel_moment, dtype)
        elif self.metric_type == MetricType.FRIEDMAN_ROBERTSON:
            return self.friedman_robertson(position, self.scale_factor, self.k, dtype)
        elif self.metric_type == MetricType.SPHERICAL_UNIVERSE:
            return self.spherical_universe(position, self.static_scale_factor, self.hubble_parameter, dtype)
        elif self.metric_type == MetricType.CYLINDRICAL_UNIVERSE:
            return self.cylindrical_universe(position, self.scale_factor, dtype)
        else:
            raise ValueError(f"Неизвестный тип метрики: {self.metric_type}")
    

# ===== Определяем статические методы для получения метрических тензоров =======================
# ==============================================================================================
    @staticmethod
    def minkowski(dtype=np.float32):
        '''Метрический тензор Минковского'''
        return np.array([
            [-1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]
            ], dtype=dtype)
    
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
            [g_tt,    0,             0,         0],
            [   0, g_rr,             0,         0],
            [   0,    0, g_theta_theta,         0],
            [   0,    0,             0, g_phi_phi]
        ], dtype=dtype)
    
    @staticmethod
    def ellis_bronnikova(vector_pos, r_0, dtype=np.float32):
        l = vector_pos[1]
        theta = vector_pos[2]

        r_min = l**2 + r_0**2
        g_tt = -1
        g_ll = 1
        g_theta_theta = r_min
        g_phi_phi = r_min * np.sin(theta)**2

        return np.array([
            [g_tt,    0,             0,         0],
            [   0, g_ll,             0,         0],
            [   0,    0, g_theta_theta,         0],
            [   0,    0,             0, g_phi_phi]
        ], dtype=dtype)
    
    @staticmethod
    def kerr_newman(vector_pos, M, Q, a, dtype=np.float64):
        r = vector_pos[1]
        theta = vector_pos[2]

        sin_theta = np.sin(theta)
        sin2_theta = sin_theta**2
        Q2 = Q**2
        a2 = a**2
        r2 = r**2
        Mr2 = 2*M*r

        delta = r2 - Mr2 + a2 + Q2
        p2 = r2 + (a*sin_theta)**2

        g_tt = -1 + (Mr2-Q2)/p2
        g_rr = p2/delta
        g_theta_theta = p2
        g_phi_phi = (r2 + a2 + (Mr2-Q2)*a2*sin2_theta/p2)*sin2_theta

        g_t_phi = - (Mr2-Q2)*a*sin2_theta/p2

        return np.array([
            [   g_tt,    0,             0,   g_t_phi],
            [      0, g_rr,             0,         0],
            [      0,    0, g_theta_theta,         0],
            [g_t_phi,    0,             0, g_phi_phi]
            ], dtype=dtype)
    
    @staticmethod
    def godel(vect_pos, a, dtype=np.float32):
        x = vect_pos[1]
        
        g_tt = -1.0
        g_ty = -np.exp(a * x)
        g_xx = 1.0
        g_yy = 0.5 * np.exp(2 * a * x)
        g_zz = 1.0

        return np.array([
            [g_tt,    0, g_ty,    0],
            [   0, g_xx,    0,    0],
            [g_ty,    0, g_yy,    0],
            [   0,    0,    0, g_zz]
        ], dtype=dtype)
    
    @staticmethod
    def friedman_robertson(vect_pos, a, k, dtype=np.float32):
        r = vect_pos[1]
        theta = vect_pos[2]
        
        # Компоненты метрического тензора
        g_tt = -1.0
        g_rr = a**2 / (1 - k * r**2)
        g_theta_theta = a**2 * r**2
        g_phi_phi = a**2 * r**2 * np.sin(theta)**2

        # Симметричный метрический тензор
        return np.array([
            [g_tt,    0,             0,         0],
            [   0, g_rr,             0,         0],
            [   0,    0, g_theta_theta,         0],
            [   0,    0,             0, g_phi_phi]
        ], dtype=dtype)
    
    @staticmethod
    def spherical_universe(vect_pos, A, H, dtype=np.float32):
        t = vect_pos[0]
        chi = vect_pos[1]
        theta = vect_pos[2]

        a = A + np.exp(H*t)
        
        # Компоненты метрического тензора
        g_tt = -1
        g_chi_chi = a**2
        g_theta_theta = a**2 * np.sin(chi)**2
        g_phi_phi = a**2 * np.sin(chi)**2 * np.sin(theta)**2

        # Симметричный метрический тензор
        return np.array([
            [g_tt,            0,             0,         0],
            [   0,    g_chi_chi,             0,         0],
            [   0,            0, g_theta_theta,         0],
            [   0,            0,             0, g_phi_phi]
        ], dtype=dtype)
    
    @staticmethod
    def cylindrical_universe(vect_pos, R, dtype=np.float32):
        theta = vect_pos[2]
        
        # Компоненты метрического тензора
        g_tt = -1
        g_nu_nu = 1
        g_theta_theta = R**2
        g_phi_phi = R**2 * np.sin(theta)**2

        # Симметричный метрический тензор
        return np.array([
            [g_tt,            0,             0,         0],
            [   0,      g_nu_nu,             0,         0],
            [   0,            0, g_theta_theta,         0],
            [   0,            0,             0, g_phi_phi]
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

    def vector_contra_to_cov(self, vect_position, vect_contra):
        g = self.get_metric_tensor(vect_position)
        # Инициализируем вектор результата контравариантный
        vect_napr_cov = np.zeros_like(vect_contra)

        if vect_contra.ndim == 1:
            for i in range(4):
                vect_napr_cov[i] = np.sum(g[i] * vect_contra)
        else:
            # Векторизованное преобразование
            vect_napr_cov = np.einsum('ij,...j->...i', g, vect_contra)
        
        return vect_napr_cov
    
    def vector_cov_to_contra(self, vect_position, vect_cov):
        g = self.get_metric_tensor(vect_position)
        g_inv = np.linalg.inv(g)
        # Инициализируем вектор результата контравариантный
        vect_napr_cov = np.zeros_like(vect_cov)

        if vect_cov.ndim == 1:
            for i in range(4):
                vect_napr_cov[i] = np.sum(g_inv[i] * vect_cov)
        else:
            # Векторизованное преобразование
            vect_napr_cov = np.einsum('ij,...j->...i', g_inv, vect_cov)
        
        return vect_napr_cov

    
    def local_to_global_vector_cont_cont(self, vector_position_cont, vect_napr_cont):
        """
        Преобразует 4-вектор из локальной ортонормированной системы отсчёта в глобальную систему координат.
        
        Параметры:
            vector_position_cont: контравариантные компоненты 4-вектора положения (t, r, θ, φ)
            vect_napr_cont: контравариантные компоненты 4-вектора в локальной системе отсчёта
            
        Возвращает:
            Контравариантные компоненты 4-вектора в глобальной системе координат
        """
        # Поддерживаем как Vector4, так и массивы NumPy
        if isinstance(vector_position_cont, Vector4):
            vector_position_cont = vector_position_cont.to_array()
        
        # Создаем копию входного вектора для безопасной работы
        local_vector = np.array(vect_napr_cont, copy=True)
        global_vector = np.zeros_like(local_vector)
        
        if self.metric_type == MetricType.MINKOWSKI:
            # Для метрики Минковского преобразование не требуется
            return local_vector
        
        elif self.metric_type == MetricType.SCHWARZSCHILD:
            r = vector_position_cont[1]
            theta = vector_position_cont[2]
            
            # Метрические коэффициенты
            gamma = 1 - self.r_s / r
            sqrt_gamma = np.sqrt(gamma)
            
            # Матрица преобразования
            if local_vector.ndim == 1:
                global_vector[0] = local_vector[0] / sqrt_gamma
                global_vector[1] = local_vector[1] * sqrt_gamma
                global_vector[2] = local_vector[2] / r
                global_vector[3] = local_vector[3] / (r * np.sin(theta))
            else:
                global_vector[..., 0] = local_vector[..., 0] / sqrt_gamma
                global_vector[..., 1] = local_vector[..., 1] * sqrt_gamma
                global_vector[..., 2] = local_vector[..., 2] / r
                global_vector[..., 3] = local_vector[..., 3] / (r * np.sin(vector_position_cont[..., 2]))
            
            return global_vector
        
        elif self.metric_type == MetricType.ELLIS_BRONNIKIVA:
            r = vector_position_cont[1]
            theta = vector_position_cont[2]
            
            # Метрические коэффициенты
            sqr_r2_r02 = np.sqrt(r**2 + self.r_0**2)
            sin_theta = np.sin(theta)
            
            # Матрица преобразования
            if local_vector.ndim == 1:
                global_vector[0] = local_vector[0]
                global_vector[1] = local_vector[1]
                global_vector[2] = local_vector[2] / sqr_r2_r02
                global_vector[3] = local_vector[3] / (sqr_r2_r02 * sin_theta)
            else:
                global_vector[..., 0] = local_vector[..., 0]
                global_vector[..., 1] = local_vector[..., 1]
                global_vector[..., 2] = local_vector[..., 2] / sqr_r2_r02
                global_vector[..., 3] = local_vector[..., 3] / (np.sqrt(vector_position_cont[..., 1]**2 + self.r_0**2) * 
                                                            np.sin(vector_position_cont[..., 2]))
            
            return global_vector
        
        elif self.metric_type == MetricType.KERR_NEWMAN:
            # Извлекаем параметры метрики
            a = self.a
            M = self.M
            Q = self.Q

            # Извлекаем координаты
            r = vector_position_cont[1]
            theta = vector_position_cont[2]

            # Вычисляем вспомогательные величины
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            a2 = a**2
            r2 = r**2

            # Вычисляем функции метрики Керра-Ньюмена
            Sigma = r2 + a2 * cos_theta**2
            Delta = r2 - 2*M*r + a2 + Q**2
            sqrt_Sigma = np.sqrt(Sigma)
            sqrt_Delta = np.sqrt(Delta)
            
            # ПРОВЕРКА: чтобы избежать деления на ноль
            if np.any(Delta <= 0) or np.any(Sigma <= 0):
                # Обработка особых случаев (горизонты)
                # Для фотонов это может быть критично
                pass

            # ИСПРАВЛЕННЫЕ компоненты тетрады
            # Временноподобный вектор
            e_hat_t_t = (r2 + a2) / (sqrt_Sigma * sqrt_Delta)
            e_hat_t_phi = a / (sqrt_Sigma * sqrt_Delta)
            
            # Пространственные векторы
            e_hat_r_r = sqrt_Delta / sqrt_Sigma
            e_hat_theta_theta = 1.0 / sqrt_Sigma
            
            # Вектор в направлении φ
            e_hat_phi_t = a * sin_theta / sqrt_Sigma
            e_hat_phi_phi = 1.0 / (sqrt_Sigma * sin_theta)

            # Выполняем преобразование
            global_vector[..., 0] = e_hat_t_t * local_vector[..., 0] + e_hat_phi_t * local_vector[..., 3]
            global_vector[..., 1] = e_hat_r_r * local_vector[..., 1]
            global_vector[..., 2] = e_hat_theta_theta * local_vector[..., 2]
            global_vector[..., 3] = e_hat_t_phi * local_vector[..., 0] + e_hat_phi_phi * local_vector[..., 3]

            return global_vector
        
        elif self.metric_type == MetricType.GOEDEL:
            # Извлекаем параметр метрики
            a = self.a
            
            # Извлекаем координаты
            x = vector_position_cont[1]
            
            # Вычисляем компоненты тетрады для метрики Гёделя
            e_hat_t_t = 1.0
            e_hat_t_y = -np.sqrt(2/3)
            
            e_hat_x_x = 1.0
            
            e_hat_y_y = np.sqrt(2/3) * np.exp(-a*x)
            
            e_hat_z_z = 1.0
            
            # Выполняем преобразование из локальной системы в глобальную
            global_vector[..., 0] = e_hat_t_t * local_vector[..., 0] + e_hat_t_y * local_vector[..., 2]
            global_vector[..., 1] = e_hat_x_x * local_vector[..., 1]
            global_vector[..., 2] = e_hat_y_y * local_vector[..., 2]
            global_vector[..., 3] = e_hat_z_z * local_vector[..., 3]
            
            return global_vector
        
        elif self.metric_type == MetricType.FRIEDMAN_ROBERTSON:
            a = self.a
            k = self.k
            r = vector_position_cont[1]
            theta = vector_position_cont[2]
            
            # Модифицированный репер для избежания комплексных чисел
            e_hat_t_t = 1.0
            
            if k == 1.0 and r >= 1.0:
                # Для r >= 1 используем аналитическое продолжение
                e_hat_r_r = np.sqrt(r**2 - 1) / a
            else:
                e_hat_r_r = np.sqrt(1 - k * r**2) / a
                
            e_hat_theta_theta = 1.0 / (a * r) if r != 0 else 0.0
            e_hat_phi_phi = 1.0 / (a * r * np.sin(theta)) if r != 0 and np.sin(theta) != 0 else 0.0
            
            # Преобразование из локальной системы в глобальную
            global_vector[..., 0] = e_hat_t_t * local_vector[..., 0]
            global_vector[..., 1] = e_hat_r_r * local_vector[..., 1]
            global_vector[..., 2] = e_hat_theta_theta * local_vector[..., 2]
            global_vector[..., 3] = e_hat_phi_phi * local_vector[..., 3]
            
            return global_vector
        
        elif self.metric_type == MetricType.SPHERICAL_UNIVERSE:
            A = self.static_scale_factor
            H = self.hubble_parameter

            t = vector_position_cont[0]
            chi = vector_position_cont[1]
            theta = vector_position_cont[2]

            a = A + np.exp(H*t)

            e_t = 1
            e_chi = 1/a 
            e_theta = 1/(a*np.sin(chi))
            e_phi = 1/(a*np.sin(chi)*np.sin(theta))

            global_vector[..., 0] = e_t * local_vector[..., 0]
            global_vector[..., 1] = e_chi * local_vector[..., 1]
            global_vector[..., 2] = e_theta * local_vector[..., 2]
            global_vector[..., 3] = e_phi * local_vector[..., 3]

            return global_vector
        
        elif self.metric_type == MetricType.CYLINDRICAL_UNIVERSE:
            R = self.scale_factor

            theta = vector_position_cont[2]

            e_t = 1
            e_nu = 1
            e_theta = 1/R
            e_phi = 1/(R*np.sin(theta))

            global_vector[..., 0] = e_t * local_vector[..., 0]
            global_vector[..., 1] = e_nu * local_vector[..., 1]
            global_vector[..., 2] = e_theta * local_vector[..., 2]
            global_vector[..., 3] = e_phi * local_vector[..., 3]

            return global_vector
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