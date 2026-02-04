#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Функция вычисления производных системы
double8 system_equations(double8 y, double r_0) {
    double l = y.s1;
    double theta = y.s2;

    double pt = y.s4;
    double pl = y.s5;
    double ptheta = y.s6;      
    double pphi = y.s7;

    double l2 = l * l;
    double r_02 = r_0 * r_0;
    double l2r_02 = l2 + r_02;
    double l2r_02_2 = l2r_02 * l2r_02;

    // Регуляризация для малых значений
    double safe_l2r_02 = fmax(l2r_02, 1e-12);
    double safe_l2r_02_2 = safe_l2r_02 * safe_l2r_02;

    double sin_theta = sin(theta);
    double sin_theta2 = sin_theta * sin_theta;
    double safe_sin_theta2 = fmax(sin_theta2, 1e-12);
    double sin_theta3 = sin_theta * safe_sin_theta2; 
    double cos_theta = cos(theta);

    double8 dy_dlambda;
    
    // Координаты
    dy_dlambda.s0 = -pt;
    dy_dlambda.s1 = pl;
    dy_dlambda.s2 = ptheta / safe_l2r_02;
    dy_dlambda.s3 = pphi / (safe_l2r_02 * safe_sin_theta2);
    
    // Импульсы
    dy_dlambda.s4 = 0.0;
    
    // Регуляризация для малых l
    double l_safe = fabs(l) < 1e-8 ? copysign(1e-8, l) : l;
    double term = l_safe / safe_l2r_02_2;
    
    dy_dlambda.s5 = term * (ptheta * ptheta + pphi * pphi / safe_sin_theta2);
    dy_dlambda.s6 = pphi * pphi * cos_theta / (safe_l2r_02 * sin_theta3);
    dy_dlambda.s7 = 0.0;
    
    return dy_dlambda;
}

// Адаптивный шаг для области горловины
double adaptive_step(double l, double pl, double r_0, double base_h) {
    double distance = fabs(l);
    double speed = fabs(pl);
    
    // Минимальный шаг у горловины
    double min_step = base_h * 0.01;
    
    // Если близко к горловине и движемся быстро - уменьшаем шаг
    if (distance < 2.0 * r_0 && speed > 0.1) {
        double factor = fmax(0.01, distance / (2.0 * r_0));
        return base_h * factor;
    }
    
    // Если проходим горловину - минимальный шаг
    if (distance < 0.5 * r_0) {
        return min_step;
    }
    
    return base_h;
}

__kernel void runge_kutta4_trajectories(
    const double r_0,
    const double h,
    const int iterations,
    const int max_points,
    const int save_step,
    __global const double8* initial_conditions,
    __global double8* trajectories,
    __global int* point_counts
) {
    const int id = get_global_id(0);
    
    double8 state = initial_conditions[id];
    double current_h = h;
    
    // Предвычисленные константы
    const double h6 = 1.0 / 6.0;
    
    // Временные переменные
    double8 k1, k2, k3, k4;
    int point_index = 0;
    int save_counter = 0;
    
    // Сохраняем начальную точку
    if (point_index < max_points) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    for (int i = 0; i < iterations; i++) {
        // Адаптивный шаг около горловины
        current_h = adaptive_step(state.s1, state.s5, r_0, h);
        const double h2 = current_h * 0.5;
        
        // Вычисление коэффициентов Рунге-Кутты
        k1 = system_equations(state, r_0);
        
        double8 temp = state + k1 * h2;
        k2 = system_equations(temp, r_0);
        
        temp = state + k2 * h2;
        k3 = system_equations(temp, r_0);
        
        temp = state + k3 * current_h;
        k4 = system_equations(temp, r_0);
        
        // Обновление решения
        state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (current_h * h6);
        
        // Сохраняем точки с заданным шагом
        save_counter++;
        if (save_counter >= save_step && point_index < max_points) {
            trajectories[id * max_points + point_index] = state;
            point_index++;
            save_counter = 0;
        }
    }
    
    // Сохраняем финальную точку, если есть место
    if (point_index < max_points) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    point_counts[id] = point_index;
}