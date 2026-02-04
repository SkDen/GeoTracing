#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

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

__kernel void runge_kutta4_trajectories(
    const double H0,        // Параметр Хаббла
    const double A,         // Начальный масштабный фактор
    const double h,         // шаг интегрирования
    const int iterations,   // количество итераций
    const int max_points,   // максимальное количество точек для сохранения
    const int save_step,    // шаг сохранения точек
    __global const double8* initial_conditions, // начальные условия
    __global double8* trajectories,             // траектории
    __global int* point_counts                  // количество точек для каждой траектории
) {
    const int id = get_global_id(0);
    
    double8 state = initial_conditions[id];
    
    // Предвычисленные константы
    const double h2 = h * 0.5;
    const double h6 = h / 6.0;
    
    // Временные переменные
    double8 k1, k2, k3, k4;
    double8 temp_state;
    int point_index = 0;
    int valid_steps = 0;
    
    // Масштабный фактор и его производная
    double a, da_dt;
    
    // Сохраняем начальную точку
    if (point_index < max_points) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    // Основной цикл интегрирования
    for (int i = 0; i < iterations; i++) {
        // Проверяем валидность состояния
        if (!is_valid_state(state)) {
            break;
        }
        
        // Вычисляем масштабный фактор для текущего времени
        compute_scale_factor(state.s0, H0, A, &a, &da_dt);
        
        // Вычисляем k1
        k1 = system_equations(state, a, da_dt);
        
        // Вычисляем k2
        temp_state = state + k1 * h2;
        compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
        k2 = system_equations(temp_state, a, da_dt);
        
        // Вычисляем k3
        temp_state = state + k2 * h2;
        compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
        k3 = system_equations(temp_state, a, da_dt);
        
        // Вычисляем k4
        temp_state = state + k3 * h;
        compute_scale_factor(temp_state.s0, H0, A, &a, &da_dt);
        k4 = system_equations(temp_state, a, da_dt);
        
        // Обновляем состояние
        state = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * h6;
        
        // Нормализуем углы после обновления
        double3 angles = normalize_angles(state.s1, state.s2, state.s3);
        state.s1 = angles.x;
        state.s2 = angles.y;
        state.s3 = angles.z;
        
        // Сохраняем состояние с заданным шагом
        valid_steps++;
        if (valid_steps % save_step == 0 && point_index < max_points) {
            trajectories[id * max_points + point_index] = state;
            point_index++;
        }
        
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