#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Функция вычисления производных системы с оптимизациями
double8 system_equations(double8 y, double r_0) {
    double l = y.s1;
    double theta = y.s2;
    double pl = y.s5;
    double ptheta = y.s6;      
    double pphi = y.s7;

    // Предварительные вычисления
    double l2 = l * l;
    double r_02 = r_0 * r_0;
    double l2r_02 = l2 + r_02;
    double l2r_02_2 = l2r_02 * l2r_02;
    
    // Регуляризация для малых значений (избегаем повторных вычислений)
    double safe_l2r_02 = fmax(l2r_02, 1e-12);
    double safe_l2r_02_2 = safe_l2r_02 * safe_l2r_02;
    
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
    dy_dlambda.s2 = ptheta / safe_l2r_02;  // dtheta/dlambda
    dy_dlambda.s3 = pphi / (safe_l2r_02 * safe_sin_theta2);  // dphi/dlambda
    
    // Импульсы
    dy_dlambda.s4 = 0.0;  // dpt/dlambda
    dy_dlambda.s7 = 0.0;  // dpphi/dlambda
    
    // Регуляризация для малых l
    double l_safe = fabs(l) < 1e-8 ? copysign(1e-8, l) : l;
    double term = l_safe / safe_l2r_02_2;
    
    dy_dlambda.s5 = term * (ptheta * ptheta + pphi * pphi / safe_sin_theta2);  // dpl/dlambda
    dy_dlambda.s6 = pphi * pphi * cos_theta / (safe_l2r_02 * sin_theta3);      // dptheta/dlambda
    
    return dy_dlambda;
}

// Адаптивный шаг для области горловины
double adaptive_step(double l, double pl, double r_0, double base_h) {
    double distance = fabs(l);
    double speed = fabs(pl);
    
    // Минимальный шаг у горловины
    double min_step = base_h * 0.01;
    
    // Уменьшаем шаг при приближении к горловине с высокой скоростью
    if (distance < 2.0 * r_0 && speed > 0.1) {
        return base_h * fmax(0.01, distance / (2.0 * r_0));
    }
    
    // Минимальный шаг вблизи горловины
    return (distance < 0.5 * r_0) ? min_step : base_h;
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
    const double h,
    const double R_sky_sphere,
    const int max_iterations,
    __global const double8* initial_conditions,
    __global double8* final_states,
    __global int* horizon_flags
) {
    const int id = get_global_id(0);
    
    double8 state = initial_conditions[id];
    double current_h = h;
    
    // Предвычисленные константы
    const double h6 = 1.0 / 6.0;
    
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
        
        // Адаптивный шаг около горловины
        current_h = adaptive_step(state.s1, state.s5, r_0, h);
        const double h2 = current_h * 0.5;
        
        // Вычисление коэффициентов Рунге-Кутты
        k1 = system_equations(state, r_0);
        
        temp_state = state + k1 * h2;
        k2 = system_equations(temp_state, r_0);
        
        temp_state = state + k2 * h2;
        k3 = system_equations(temp_state, r_0);
        
        temp_state = state + k3 * current_h;
        k4 = system_equations(temp_state, r_0);
        
        // Обновление решения
        state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (current_h * h6);
        
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