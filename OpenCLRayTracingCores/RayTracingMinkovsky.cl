#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// Безопасное вычисление обратного значения с защитой от деления на ноль
inline double safe_divide(double numerator, double denominator, double epsilon) {
    return denominator != 0.0 ? numerator / (denominator + copysign(epsilon, denominator)) : 0.0;
}

// Функция для нормализации угла theta в диапазон [0, pi]
double normalize_theta(double theta) {
    theta = fmod(theta, 2.0 * M_PI);
    if (theta < 0.0) theta += 2.0 * M_PI;
    
    // Отображаем углы больше pi в правильный диапазон
    if (theta > M_PI) {
        theta = 2.0 * M_PI - theta;
        // Здесь может потребоваться изменение знака импульса p_theta
        // для сохранения физической корректности
    }
    
    return theta;
}

// Функция вычисления производных системы с улучшенной обработкой сингулярностей
double8 system_equations(double8 y) {
    double t = y.s0;
    double r = y.s1;
    double theta = y.s2;
    double phi = y.s3;
    
    double p_t = y.s4;
    double p_r = y.s5;
    double p_theta = y.s6;
    double p_phi = y.s7;
    
    // Малая константа для регуляризации сингулярностей
    const double epsilon = 1e-12;
    
    // Нормализуем угол theta для избежания сингулярностей
    // theta = normalize_theta(theta);
    
    double r2 = r * r;
    double r3 = r2 * r;
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    
    // Регуляризация sin(theta) для избежания деления на ноль
    double sin_theta_safe = sin_theta + copysign(epsilon, sin_theta);
    double sin2_theta_safe = sin_theta_safe * sin_theta_safe;
    double sin3_theta_safe = sin2_theta_safe * sin_theta_safe;
    
    double8 dy_dlambda;
    
    // Уравнения для координат (производные от гамильтониана по импульсам)
    dy_dlambda.s0 = -p_t;
    dy_dlambda.s1 = p_r;
    dy_dlambda.s2 = safe_divide(p_theta, r2, epsilon);
    dy_dlambda.s3 = safe_divide(p_phi, r2 * sin2_theta_safe, epsilon);
    
    // Уравнения для импульсов (отрицательные производные от гамильтониана по координатам)
    dy_dlambda.s4 = 0.0;
    dy_dlambda.s5 = safe_divide(p_theta * p_theta, r3, epsilon) + 
                    safe_divide(p_phi * p_phi, r3 * sin2_theta_safe, epsilon);
    dy_dlambda.s6 = safe_divide(p_phi * p_phi * cos_theta, r2 * sin3_theta_safe, epsilon);
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

__kernel void runge_kutta4_tracing(
    const double h,
    const double R_sky_sphere,
    const int max_iterations,
    __global const double8* initial_conditions,
    __global double8* point_status,
    __global int* horizon_flag
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
    
    for (int i = 0; i < max_iterations; i++) {
        // Проверяем валидность состояния
        if (!is_valid_state(state)) {
            point_status[id] = state;
            horizon_flag[id] = 3;  // Флаг невалидного состояния
            return;
        }
        
        // Вычисляем коэффициенты Рунге-Кутты
        k1 = system_equations(state);
        
        temp_state = state + k1 * h2;
        k2 = system_equations(temp_state);
        
        temp_state = state + k2 * h2;
        k3 = system_equations(temp_state);
        
        temp_state = state + k3 * h;
        k4 = system_equations(temp_state);
        
        // Обновляем состояние
        state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * h6;

        // Определяем пересечение небесной сферы
        if (state.s1 >= R_sky_sphere) {
            point_status[id] = state;
            horizon_flag[id] = 0;
            return;
        }   
    }

    // В случае превышения количества итераций
    // Фотон движется по орбите ЧД
    point_status[id] = state;
    horizon_flag[id] = 2;
}