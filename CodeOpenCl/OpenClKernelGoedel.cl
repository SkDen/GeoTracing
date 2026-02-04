#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Функция вычисления производных системы для метрики Гёделя
double8 system_equations(double8 y, double a) {
    // Извлекаем переменные состояния
    double t = y.s0;
    double x = y.s1;
    double y_coord = y.s2;
    double z = y.s3;
    double pt = y.s4;
    double px = y.s5;
    double py = y.s6;
    double pz = y.s7;
    
    double8 dy_dlambda;
    
    // Уравнения для координат
    dy_dlambda.s0 = -1.0/3.0 * pt - 2.0/3.0 * exp(-a*x) * py;
    dy_dlambda.s1 = px;
    dy_dlambda.s2 = -2.0/3.0 * exp(-a*x) * pt + 2.0/3.0 * exp(-2.0*a*x) * py;
    dy_dlambda.s3 = pz;
    
    // Уравнения для импульсов
    dy_dlambda.s4 = 0.0;
    dy_dlambda.s5 = (2.0*a/3.0) * exp(-a*x) * pt * py - 
                    (2.0*a/3.0) * exp(-2.0*a*x) * py * py;
    dy_dlambda.s6 = 0.0;
    dy_dlambda.s7 = 0.0;
    
    return dy_dlambda;
}

__kernel void runge_kutta4_trajectories(
    const double a,
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
        k1 = system_equations(state, a);
        
        double8 temp = state + k1 * h2;
        k2 = system_equations(temp, a);
        
        temp = state + k2 * h2;
        k3 = system_equations(temp, a);
        
        temp = state + k3 * h;
        k4 = system_equations(temp, a);
        
        // Обновление решения
        state += (k1 + 2.0*k2 + 2.0*k3 + k4) * h6;
        
        // Сохраняем точки с заданным шагом
        if((i % save_step) == 0 && point_index < max_points) {
            trajectories[id * max_points + point_index] = state;
            point_index++;
        }
    }
    
    // Сохраняем финальную точку, если есть место
    if(point_index < max_points) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    point_counts[id] = point_index;
}