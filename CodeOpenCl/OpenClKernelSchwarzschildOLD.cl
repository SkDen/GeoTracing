#pragma OPENCL EXTENSION cl_khr_fp64 : enable


// Функция вычисления производных системы
double8 system_equations(double8 y, double r_s) {
    double r = y.s1;
    double theta = y.s2;
    double pr = y.s5;
    double ptheta = y.s6;
    double pphi = y.s7;
    
    double r2 = r * r;
    double r3 = r2 * r;
    double sin_theta = sin(theta);
    double sin2_theta = sin_theta * sin_theta;
    double sin3_theta = sin2_theta * sin_theta;
    double cos_theta = cos(theta);
    
    double gamma = 1.0 - r_s / r;
    double gamma_diff = r_s / r2;
    double gamma_inv_diff = -r_s / ((r - r_s) * (r - r_s));
    
    double8 dy_dlambda;
    
    // Координаты
    dy_dlambda.s0 = - y.s4 / gamma;
    dy_dlambda.s1 = gamma * pr;
    dy_dlambda.s2 = ptheta / r2;
    dy_dlambda.s3 = pphi / (r2 * sin2_theta);
    

    // Импульсы
    dy_dlambda.s4 = 0.0;
    dy_dlambda.s5 = gamma_inv_diff * y.s4 * y.s4 * 0.5
                    - gamma_diff * pr * pr * 0.5
                    + (ptheta * ptheta) / r3
                    + (pphi * pphi) / (r3 * sin2_theta);
    dy_dlambda.s6 = (pphi * pphi) * cos_theta / (r2 * sin3_theta);
    dy_dlambda.s7 = 0.0;
    
        
    
    return dy_dlambda;
}

__kernel void runge_kutta4_trajectories(
    const double r_s,
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
    
    // Предвычисленные константы
    const double h2 = h * 0.5;
    const double h6 = h / 6.0;
    
    // Временные переменные
    double8 k1, k2, k3, k4;
    int point_index = 0;
    
    // Сохраняем начальную точку
    if(point_index < max_points) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    for(int i = 0; i < iterations; i++) {
        // Вычисление коэффициентов Рунге-Кутты
        k1 = system_equations(state, r_s);
        
        double8 temp = state + k1 * h2;
        k2 = system_equations(temp, r_s);
        
        temp = state + k2 * h2;
        k3 = system_equations(temp, r_s);
        
        temp = state + k3 * h;
        k4 = system_equations(temp, r_s);
        
        // Обновление решения
        state += (k1 + 2.0*k2 + 2.0*k3 + k4) * h6;
        
        // Проверка на выход за горизонт событий
        if(state.s1 < 1.01 * r_s) {
            // Сохраняем последнюю точку перед выходом
            if(point_index < max_points) {
                trajectories[id * max_points + point_index] = state;
                point_index++;
            }
            break;
        }
        
        // Сохраняем точки с заданным шагом
        if((i % save_step) == 0 && point_index < max_points) {
            trajectories[id * max_points + point_index] = state;
            point_index++;
        }
    }
    
    // Сохраняем финальную точку, если есть место
    if(point_index < max_points && state.s1 >= 1.01 * r_s) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    point_counts[id] = point_index;
}