#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Вспомогательная функция для вычисления r(l)
double r_computation(double l, double r_0, double a, double m) {
    double arg1 = m * (l - a);
    double arg2 = m * (l + a);
    double arg3 = m * a;
    
    // Ограничение аргументов для избежания переполнения
    arg1 = clamp(arg1, -500.0, 500.0);
    arg2 = clamp(arg2, -500.0, 500.0);
    arg3 = clamp(arg3, -500.0, 500.0);
    
    double cosh1 = cosh(arg1);
    double cosh2 = cosh(arg2);
    double cosh3 = cosh(arg3);
    
    return r_0 + 1.0/(2.0*m) * log( (cosh1 * cosh2) / (cosh3 * cosh3) );
}

// Вспомогательная функция для вычисления r'(l)
double r_prime_computation(double l, double a, double m) {
    double arg1 = m * (l - a);
    double arg2 = m * (l + a);
    
    // Ограничение аргументов для избежания переполнения
    arg1 = clamp(arg1, -500.0, 500.0);
    arg2 = clamp(arg2, -500.0, 500.0);
    
    return 0.5 * (tanh(arg1) + tanh(arg2));
}

// Функция вычисления производных системы
double8 system_equations(double8 y, double r_0, double a, double m) {
    double l = y.s1;
    double theta = y.s2;
    double pl = y.s5;
    double ptheta = y.s6;      
    double pphi = y.s7;

    // Вычисляем r(l) и r'(l)
    double r_val = r_computation(l, r_0, a, m);
    double r_prime_val = r_prime_computation(l, a, m);

    double r2 = r_val * r_val;
    double safe_r2 = fmax(r2, 1e-12);
    double safe_r_val = fmax(fabs(r_val), 1e-12);

    // Тригонометрические вычисления
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double sin_theta2 = sin_theta * sin_theta;
    double safe_sin_theta2 = fmax(sin_theta2, 1e-12);
    double sin_theta3 = sin_theta * safe_sin_theta2; 

    double8 dy_dlambda;
    
    // Координаты
    dy_dlambda.s0 = -y.s4;  // dt/dlambda
    dy_dlambda.s1 = pl;     // dl/dlambda
    dy_dlambda.s2 = ptheta / safe_r2;  // dtheta/dlambda
    dy_dlambda.s3 = pphi / (safe_r2 * safe_sin_theta2);  // dphi/dlambda
    
    // Импульсы
    dy_dlambda.s4 = 0.0;  // dpt/dlambda
    dy_dlambda.s7 = 0.0;  // dpphi/dlambda
    
    // Вычисляем term = r' / r^3
    double term = r_prime_val / (safe_r2 * safe_r_val);
    
    dy_dlambda.s5 = term * (ptheta * ptheta + pphi * pphi / safe_sin_theta2);  // dpl/dlambda
    dy_dlambda.s6 = pphi * pphi * cos_theta / (safe_r2 * sin_theta3);      // dptheta/dlambda
    
    return dy_dlambda;
}

// Проверка валидности состояния (без изменений)
bool is_valid_state(double8 state) {
    return !(isnan(state.s0) || isnan(state.s1) || isnan(state.s2) || isnan(state.s3) ||
             isnan(state.s4) || isnan(state.s5) || isnan(state.s6) || isnan(state.s7) ||
             isinf(state.s0) || isinf(state.s1) || isinf(state.s2) || isinf(state.s3) ||
             isinf(state.s4) || isinf(state.s5) || isinf(state.s6) || isinf(state.s7));
}

__kernel void runge_kutta4_tracing(
    const double r_0,
    const double a,
    const double m,
    const double h,
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
        k1 = system_equations(state, r_0, a, m);
        
        temp_state = state + k1 * h_half;
        k2 = system_equations(temp_state, r_0, a, m);
        
        temp_state = state + k2 * h_half;
        k3 = system_equations(temp_state, r_0, a, m);
        
        temp_state = state + k3 * h;
        k4 = system_equations(temp_state, r_0, a, m);
        
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