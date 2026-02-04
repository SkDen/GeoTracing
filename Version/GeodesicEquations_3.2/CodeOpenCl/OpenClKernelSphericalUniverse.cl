#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// Константы для адаптивного шага
#define SAFETY_FACTOR 0.9
#define MIN_SCALE_FACTOR 0.2
#define MAX_SCALE_FACTOR 5.0
#define TOLERANCE 1e-8
#define MAX_ITERATIONS 1000000

// Функция для вычисления масштабного фактора a(t) и его производной da/dt
void compute_scale_factor(double t, double H0, double A, double* a, double* da_dt) {
    // Ограничиваем время для избежания переполнения
    double t_limited = t;
    if (t_limited > 1000.0) {
        t_limited = 1000.0;
    }
    
    // Экспоненциальное расширение
    *a = A * exp(H0 * t_limited);
    *da_dt = H0 * A * exp(H0 * t_limited);
}

// Функция для нормализации углов в правильные диапазоны
double3 normalize_angles(double chi, double theta, double phi) {
    double3 normalized;
    
    // Нормализация chi в диапазон [0, pi]
    normalized.x = fmod(chi, 2.0 * M_PI);
    if (normalized.x < 0.0) normalized.x += 2.0 * M_PI;
    if (normalized.x > M_PI) normalized.x = 2.0 * M_PI - normalized.x;
    
    // Нормализация theta в диапазон [0, pi]
    normalized.y = fmod(theta, 2.0 * M_PI);
    if (normalized.y < 0.0) normalized.y += 2.0 * M_PI;
    if (normalized.y > M_PI) normalized.y = 2.0 * M_PI - normalized.y;
    
    // Нормализация phi в диапазон [0, 2*pi]
    normalized.z = fmod(phi, 2.0 * M_PI);
    if (normalized.z < 0.0) normalized.z += 2.0 * M_PI;
    
    return normalized;
}

// Функция вычисления производных системы для сферической вселенной в гиперсферических координатах
double8 system_equations(double8 y, double a, double da_dt) {
    // Извлекаем переменные состояния
    double t = y.s0;
    double chi = y.s1;
    double theta = y.s2;
    double phi = y.s3;
    
    double p_t = y.s4;
    double p_chi = y.s5;
    double p_theta = y.s6;
    double p_phi = y.s7;
    
    // Нормализуем углы для избежания сингулярностей
    double3 angles = normalize_angles(chi, theta, phi);
    chi = angles.x;
    theta = angles.y;
    phi = angles.z;
    
    // Предварительные вычисления с защитой от деления на ноль
    double sin_chi = sin(chi);
    double sin_theta = sin(theta);
    
    // Регуляризация: добавляем малую константу чтобы избежать деления на ноль
    double sin_chi_reg = sin_chi + copysign(1e-12, sin_chi);
    double sin_theta_reg = sin_theta + copysign(1e-12, sin_theta);
    
    double sin2_chi_reg = sin_chi_reg * sin_chi_reg;
    double sin3_chi_reg = sin2_chi_reg * sin_chi_reg;
    double cos_chi = cos(chi);
    
    double sin2_theta_reg = sin_theta_reg * sin_theta_reg;
    double sin3_theta_reg = sin2_theta_reg * sin_theta_reg;
    double cos_theta = cos(theta);
    
    double a2 = a * a;
    double a3 = a2 * a;
    
    double8 dy_dlambda;
    
    // Уравнения для координат
    dy_dlambda.s0 = -p_t;
    dy_dlambda.s1 = p_chi / a2;
    dy_dlambda.s2 = p_theta / (a2 * sin2_chi_reg);
    dy_dlambda.s3 = p_phi / (a2 * sin2_chi_reg * sin2_theta_reg);
    
    // Уравнения для импульсов
    dy_dlambda.s4 = da_dt / a3 * (p_chi * p_chi + 
                    p_theta * p_theta / sin2_chi_reg + 
                    p_phi * p_phi / (sin2_chi_reg * sin2_theta_reg));
    
    // Защита от сингулярностей в уравнениях для импульсов
    if (fabs(sin_chi_reg) > 1e-8 && fabs(sin_theta_reg) > 1e-8) {
        dy_dlambda.s5 = (cos_chi / (a2 * sin3_chi_reg)) * 
                       (p_theta * p_theta + 
                        p_phi * p_phi / sin2_theta_reg);
        dy_dlambda.s6 = (cos_theta / (a2 * sin2_chi_reg * sin3_theta_reg)) * 
                       p_phi * p_phi;
    } else {
        // Если близко к сингулярности, используем приближенные формулы
        dy_dlambda.s5 = 0.0;
        dy_dlambda.s6 = 0.0;
    }
    
    dy_dlambda.s7 = 0.0;
    
    return dy_dlambda;
}

// Функция для проверки на NaN и бесконечности
bool is_valid_state(double8 state) {
    return !(isnan(state.s0) || isnan(state.s1) || isnan(state.s2) || isnan(state.s3) ||
             isnan(state.s4) || isnan(state.s5) || isnan(state.s6) || isnan(state.s7) ||
             isinf(state.s0) || isinf(state.s1) || isinf(state.s2) || isinf(state.s3) ||
             isinf(state.s4) || isinf(state.s5) || isinf(state.s6) || isinf(state.s7));
}

// Функция для выполнения одного шага Рунге-Кутты 4-го порядка
double8 rk4_step(double8 state, double h, double H0, double A) {
    double a, da_dt;
    
    // Вычисляем масштабный фактор для текущего времени
    compute_scale_factor(state.s0, H0, A, &a, &da_dt);
    
    // Вычисляем k1
    double8 k1 = system_equations(state, a, da_dt);
    
    // Вычисляем k2
    double8 temp_state = state + k1 * (h * 0.5);
    compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
    double8 k2 = system_equations(temp_state, a, da_dt);
    
    // Вычисляем k3
    temp_state = state + k2 * (h * 0.5);
    compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
    double8 k3 = system_equations(temp_state, a, da_dt);
    
    // Вычисляем k4
    temp_state = state + k3 * h;
    compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
    double8 k4 = system_equations(temp_state, a, da_dt);
    
    // Обновляем состояние
    return state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (h / 6.0);
}

// Функция для выполнения одного шага Рунге-Кутты 5-го порядка (для оценки ошибки)
double8 rk5_step(double8 state, double h, double H0, double A) {
    double a, da_dt;
    
    // Вычисляем масштабный фактор для текущего времени
    compute_scale_factor(state.s0, H0, A, &a, &da_dt);
    
    // Вычисляем k1
    double8 k1 = system_equations(state, a, da_dt);
    
    // Вычисляем k2
    double8 temp_state = state + k1 * (h * 0.25);
    compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
    double8 k2 = system_equations(temp_state, a, da_dt);
    
    // Вычисляем k3
    temp_state = state + (k1 * 3.0 + k2 * 9.0) * (h / 32.0);
    compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
    double8 k3 = system_equations(temp_state, a, da_dt);
    
    // Вычисляем k4
    temp_state = state + (k1 * 1932.0 - k2 * 7200.0 + k3 * 7296.0) * (h / 2197.0);
    compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
    double8 k4 = system_equations(temp_state, a, da_dt);
    
    // Вычисляем k5
    temp_state = state + (k1 * 439.0 / 216.0 - k2 * 8.0 + k3 * 3680.0 / 513.0 - k4 * 845.0 / 4104.0) * h;
    compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
    double8 k5 = system_equations(temp_state, a, da_dt);
    
    // Вычисляем k6
    temp_state = state + (-k1 * 8.0 / 27.0 + k2 * 2.0 - k3 * 3544.0 / 2565.0 + k4 * 1859.0 / 4104.0 - k5 * 11.0 / 40.0) * h;
    compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
    double8 k6 = system_equations(temp_state, a, da_dt);
    
    // Обновляем состояние (метод 5-го порядка)
    return state + (k1 * 16.0 / 135.0 + k3 * 6656.0 / 12825.0 + k4 * 28561.0 / 56430.0 - k5 * 9.0 / 50.0 + k6 * 2.0 / 55.0) * h;
}

// Функция для вычисления ошибки между двумя решениями
double compute_error(double8 y1, double8 y2) {
    double error = 0.0;
    
    // Вычисляем относительную ошибку для каждой компоненты
    error += fabs(y1.s0 - y2.s0) / (1.0 + fabs(y1.s0));
    error += fabs(y1.s1 - y2.s1) / (1.0 + fabs(y1.s1));
    error += fabs(y1.s2 - y2.s2) / (1.0 + fabs(y1.s2));
    error += fabs(y1.s3 - y2.s3) / (1.0 + fabs(y1.s3));
    error += fabs(y1.s4 - y2.s4) / (1.0 + fabs(y1.s4));
    error += fabs(y1.s5 - y2.s5) / (1.0 + fabs(y1.s5));
    error += fabs(y1.s6 - y2.s6) / (1.0 + fabs(y1.s6));
    error += fabs(y1.s7 - y2.s7) / (1.0 + fabs(y1.s7));
    
    return error / 8.0;  // Средняя относительная ошибка
}

__kernel void runge_kutta4_trajectories(
    const double H0,        // Параметр Хаббла
    const double A,         // Начальный масштабный фактор
    const double h0,        // начальный шаг интегрирования
    const int iteration0,
    const int max_points,   // максимальное количество точек для сохранения
    const int save_step,    // шаг сохранения точек
    __global const double8* initial_conditions, // начальные условия
    __global double8* trajectories,             // траектории
    __global int* point_counts                  // количество точек для каждой траектории
) {
    const int id = get_global_id(0);
    
    double lambda_end = h0 * iteration0;

    double8 state = initial_conditions[id];
    double lambda = 0.0;
    double h = h0;
    
    int point_index = 0;
    int iteration = 0;
    
    // Сохраняем начальную точку
    if (point_index < max_points) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    // Основной цикл интегрирования
    while (lambda < lambda_end && iteration < MAX_ITERATIONS) {
        // Проверяем валидность состояния
        if (!is_valid_state(state)) {
            break;
        }
        
        // Выполняем шаг Рунге-Кутты 4-го и 5-го порядка
        double8 state_rk4 = rk4_step(state, h, H0, A);
        double8 state_rk5 = rk5_step(state, h, H0, A);
        
        // Вычисляем ошибку
        double error = compute_error(state_rk4, state_rk5);
        
        // Вычисляем коэффициент масштабирования шага
        double scale = SAFETY_FACTOR * pow(TOLERANCE / error, 0.2);
        scale = fmax(MIN_SCALE_FACTOR, fmin(scale, MAX_SCALE_FACTOR));
        
        if (error <= TOLERANCE) {
            // Шг принят, используем решение более высокого порядка (RK5)
            state = state_rk5;
            lambda += h;
            
            // Нормализуем углы после обновления
            double3 angles = normalize_angles(state.s1, state.s2, state.s3);
            state.s1 = angles.x;
            state.s2 = angles.y;
            state.s3 = angles.z;
            
            // Сохраняем состояние с заданным шагом
            if (iteration % save_step == 0 && point_index < max_points) {
                trajectories[id * max_points + point_index] = state;
                point_index++;
            }
        }
        
        // Корректируем шаг для следующей итерации
        h *= scale;
        
        // Гарантируем, что не выйдем за пределы lambda_end
        if (lambda + h > lambda_end) {
            h = lambda_end - lambda;
        }
        
        iteration++;
        
        // Проверяем выход за пределы разумных значений
        if (fabs(state.s0) > 1e6 || fabs(state.s1) > 2.0 * M_PI || 
            fabs(state.s2) > 2.0 * M_PI || fabs(state.s3) > 4.0 * M_PI ||
            fabs(state.s4) > 1e6 || fabs(state.s5) > 1e6 || 
            fabs(state.s6) > 1e6 || fabs(state.s7) > 1e6) {
            break;
        }
    }
    
    // Сохраняем количество записанных точек
    point_counts[id] = point_index;
}