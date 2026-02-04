#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Безопасное вычисление тригонометрических функций
inline double safe_sin(double theta) {
    return sin(fmax(fmin(theta, M_PI - 1e-12), 1e-12));
}

inline double safe_cos(double theta) {
    return cos(fmax(fmin(theta, M_PI - 1e-12), 1e-12));
}

inline double safe_divide(double numerator, double denominator) {
    return denominator != 0.0 ? numerator / denominator : copysign(1e12, numerator) * copysign(1.0, denominator);
}

// Функция вычисления производных системы
double8 system_equations(double8 y, double r_0) {
    double l = y.s1;
    double theta = y.s2;
    double pl = y.s5;
    double ptheta = y.s6;      
    double pphi = y.s7;

    // Регуляризация угла theta
    theta = fmax(fmin(theta, M_PI - 1e-12), 1e-12);
    
    // Предварительные вычисления с регуляризацией
    double l2 = l * l;
    double r_02 = r_0 * r_0;
    double l2r_02 = l2 + r_02;
    
    // Безопасные вычисления
    double safe_l2r_02 = fmax(l2r_02, 1e-12);
    double sin_theta = safe_sin(theta);
    double cos_theta = safe_cos(theta);
    double sin_theta2 = sin_theta * sin_theta;
    double safe_sin_theta2 = fmax(sin_theta2, 1e-12);
    
    double8 dy_dlambda;
    
    // Координаты
    dy_dlambda.s0 = -y.s4;
    dy_dlambda.s1 = pl;
    dy_dlambda.s2 = safe_divide(ptheta, safe_l2r_02);
    dy_dlambda.s3 = safe_divide(pphi, safe_l2r_02 * safe_sin_theta2);
    
    // Импульсы
    dy_dlambda.s4 = 0.0;
    dy_dlambda.s7 = 0.0;
    
    // Безопасное вычисление для производных импульсов
    if (fabs(l) < 1e-8) {
        // Специальная обработка для малых l
        dy_dlambda.s5 = 0.0;
        dy_dlambda.s6 = safe_divide(pphi * pphi * cos_theta, 
                                   safe_l2r_02 * sin_theta * safe_sin_theta2);
    } else {
        double term = l / (safe_l2r_02 * safe_l2r_02);
        dy_dlambda.s5 = term * (ptheta * ptheta + safe_divide(pphi * pphi, safe_sin_theta2));
        dy_dlambda.s6 = safe_divide(pphi * pphi * cos_theta, 
                                   safe_l2r_02 * sin_theta * safe_sin_theta2);
    }
    
    return dy_dlambda;
}

// Проверка валидности состояния
bool is_valid_state(double8 state) {
    return !(isnan(state.s0) || isnan(state.s1) || isnan(state.s2) || isnan(state.s3) ||
             isnan(state.s4) || isnan(state.s5) || isnan(state.s6) || isnan(state.s7) ||
             isinf(state.s0) || isinf(state.s1) || isinf(state.s2) || isinf(state.s3) ||
             isinf(state.s4) || isinf(state.s5) || isinf(state.s6) || isinf(state.s7));
}

__kernel void runge_kutta4_tracing(
    const double r_0,
    const double h,  // Фиксированный шаг интегрирования
    const double R_sky_sphere,
    const int max_iterations,
    __global const double8* initial_conditions,
    __global double8* final_states,
    __global int* horizon_flags
) {
    const int id = get_global_id(0);
    
    double8 state = initial_conditions[id];
    
    // Предвычисленные константы
    const double h_half = h * 0.5;
    const double h_sixth = h / 6.0;
    
    // Временные переменные
    double8 k1, k2, k3, k4;
    double8 temp_state;
    
    for (int i = 0; i < max_iterations; i++) {
        // Проверка валидности состояния
        if (!is_valid_state(state)) {
            final_states[id] = state;
            horizon_flags[id] = 3;  // Невалидное состояние
            return;
        }
        
        // Вычисление коэффициентов Рунге-Кутты с фиксированным шагом
        k1 = system_equations(state, r_0);
        
        temp_state = state + k1 * h_half;
        k2 = system_equations(temp_state, r_0);
        
        temp_state = state + k2 * h_half;
        k3 = system_equations(temp_state, r_0);
        
        temp_state = state + k3 * h;
        k4 = system_equations(temp_state, r_0);
        
        // Обновление решения
        state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * h_sixth;
        
        // Проверка достижения небесной сферы
        if (state.s1 >= R_sky_sphere) {
            final_states[id] = state;
            horizon_flags[id] = 0;  // Достигнута небесная сфера
            return;
        }
        
        // Проверка на другую вселенную
        if (state.s1 <= -R_sky_sphere) {
            final_states[id] = state;
            horizon_flags[id] = 4;  // Небесная сфера другой вселенной
            return;
        }
    }
    
    // Превышено максимальное количество итераций
    final_states[id] = state;
    horizon_flags[id] = 2;
}