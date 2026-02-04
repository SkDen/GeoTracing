#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Функция вычисления производных системы для сферической вселенной
double8 system_equations(double8 y, double a, double k) {
    // Извлекаем переменные состояния
    double r = y.s1;
    double theta = y.s2;
    double p_r = y.s5;
    double p_theta = y.s6;
    double p_phi = y.s7;
    
    // Предварительные вычисления
    double r2 = r * r;
    double r3 = r2 * r;
    double sin_theta = sin(theta);
    double sin2_theta = sin_theta * sin_theta;
    double sin3_theta = sin2_theta * sin_theta;
    double cos_theta = cos(theta);
    
    double denom_r = 1.0 - k * r2;
    double a2 = a * a;
    
    double8 dy_dlambda;
    
    // Уравнения для координат
    dy_dlambda.s0 = -y.s4;
    dy_dlambda.s1 = denom_r / a2 * p_r;
    dy_dlambda.s2 = 1.0 / (a2 * r2) * p_theta;
    dy_dlambda.s3 = 1.0 / (a2 * r2 * sin2_theta) * p_phi;
    
    // Уравнения для импульсов
    dy_dlambda.s4 = 0.0;
    dy_dlambda.s5 = (k * r / a2) * p_r * p_r + 
                    (1.0 / (a2 * r3)) * (p_theta * p_theta + 
                    p_phi * p_phi / sin2_theta);
    dy_dlambda.s6 = (cos_theta / (a2 * r2 * sin3_theta)) * 
                    p_phi * p_phi;
    dy_dlambda.s7 = 0.0;
    
    return dy_dlambda;
}

__kernel void runge_kutta4_trajectories(
    const double a,         // масштабный фактор
    const double k,         // параметр кривизны
    const double h,         // шаг интегрирования
    const int iterations,   // количество итераций
    const int max_points,   // максимальное количество точек для сохранения
    const int save_step,    // шаг сохранения точек
    __global const double8* initial_conditions,
    __global double8* trajectories,
    __global int* point_counts
) {
    const int id = get_global_id(0);
    
    double8 state = initial_conditions[id];
    
    // Предвычисленные константы
    const double h2 = h * 0.5;
    const double h6 = h / 6.0;
    
    double8 k1, k2, k3, k4;
    int point_index = 0;
    
    // Сохраняем начальную точку
    if(point_index < max_points) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    for(int i = 0; i < iterations; i++) {
        // Вычисление коэффициентов Рунге-Кутты
        k1 = system_equations(state, a, k);
        
        double8 temp = state + k1 * h2;
        k2 = system_equations(temp, a, k);
        
        temp = state + k2 * h2;
        k3 = system_equations(temp, a, k);
        
        temp = state + k3 * h;
        k4 = system_equations(temp, a, k);
        
        // Обновление решения
        state += (k1 + 2.0*k2 + 2.0*k3 + k4) * h6;
        
        // Проверка на особые точки
        if(state.s1 < 1e-10 || fabs(sin(state.s2)) < 1e-10) {
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
    if(point_index < max_points && state.s1 >= 1e-10 && fabs(sin(state.s2)) >= 1e-10) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    point_counts[id] = point_index;
}