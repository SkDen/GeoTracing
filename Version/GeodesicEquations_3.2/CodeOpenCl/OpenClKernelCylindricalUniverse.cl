#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// Функция для нормализации углов
double2 normalize_angles(double theta, double phi) {
    double2 normalized;
    
    // Нормализация theta
    theta = fmod(theta, 2.0 * M_PI);
    if (theta < 0.0) theta += 2.0 * M_PI;
    
    if (theta > M_PI) {
        theta = 2.0 * M_PI - theta;
        phi += M_PI;
    }
    
    // Нормализация phi
    phi = fmod(phi, 2.0 * M_PI);
    if (phi < 0.0) phi += 2.0 * M_PI;
    
    normalized.x = theta;
    normalized.y = phi;
    return normalized;
}

// Функция вычисления производных
double8 system_equations(double8 y, double R) {
    double t = y.s0;
    double nu = y.s1;
    double theta = y.s2;
    double phi = y.s3;
    
    double p_t = y.s4;
    double p_nu = y.s5;
    double p_theta = y.s6;
    double p_phi = y.s7;
    
    double R2 = R * R;
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    
    // Регуляризация сингулярностей
    double sin_theta_reg = sin_theta;
    if (fabs(sin_theta_reg) < 1e-8) {
        sin_theta_reg = copysign(1e-8, sin_theta);
    }
    
    double sin2_theta_reg = sin_theta_reg * sin_theta_reg;
    double sin3_theta_reg = sin2_theta_reg * sin_theta_reg;
    
    double8 dy_dlambda;
    
    // Уравнения для координат
    dy_dlambda.s0 = -p_t;
    dy_dlambda.s1 = p_nu;
    dy_dlambda.s2 = p_theta / R2;
    dy_dlambda.s3 = p_phi / (R2 * sin2_theta_reg);
    
    // Уравнения для импульсов
    dy_dlambda.s4 = 0.0;
    dy_dlambda.s5 = 0.0;
    dy_dlambda.s6 = (p_phi * p_phi * cos_theta) / (R2 * sin3_theta_reg);
    dy_dlambda.s7 = 0.0;
    
    return dy_dlambda;
}

// Функция проверки состояния
bool is_valid_state(double8 state) {
    return !(isnan(state.s0) || isnan(state.s1) || isnan(state.s2) || isnan(state.s3) ||
             isnan(state.s4) || isnan(state.s5) || isnan(state.s6) || isnan(state.s7) ||
             isinf(state.s0) || isinf(state.s1) || isinf(state.s2) || isinf(state.s3) ||
             isinf(state.s4) || isinf(state.s5) || isinf(state.s6) || isinf(state.s7));
}

// Ядро для вычисления траекторий
__kernel void runge_kutta4_trajectories(
    const double R,
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
    double8 k1, k2, k3, k4;
    double8 temp_state;
    
    int point_index = 0;
    int valid_steps = 0;
    
    // Сохранение начальной точки
    if (point_index < max_points) {
        trajectories[id * max_points + point_index] = state;
        point_index++;
    }
    
    // Основной цикл интегрирования
    for (int i = 0; i < iterations; i++) {
        if (!is_valid_state(state)) break;
        
        // Вычисление k1
        k1 = system_equations(state, R);
        
        // Вычисление k2
        temp_state.s0 = state.s0 + k1.s0 * h * 0.5;
        temp_state.s1 = state.s1 + k1.s1 * h * 0.5;
        temp_state.s2 = state.s2 + k1.s2 * h * 0.5;
        temp_state.s3 = state.s3 + k1.s3 * h * 0.5;
        temp_state.s4 = state.s4 + k1.s4 * h * 0.5;
        temp_state.s5 = state.s5 + k1.s5 * h * 0.5;
        temp_state.s6 = state.s6 + k1.s6 * h * 0.5;
        temp_state.s7 = state.s7 + k1.s7 * h * 0.5;
        k2 = system_equations(temp_state, R);
        
        // Вычисление k3
        temp_state.s0 = state.s0 + k2.s0 * h * 0.5;
        temp_state.s1 = state.s1 + k2.s1 * h * 0.5;
        temp_state.s2 = state.s2 + k2.s2 * h * 0.5;
        temp_state.s3 = state.s3 + k2.s3 * h * 0.5;
        temp_state.s4 = state.s4 + k2.s4 * h * 0.5;
        temp_state.s5 = state.s5 + k2.s5 * h * 0.5;
        temp_state.s6 = state.s6 + k2.s6 * h * 0.5;
        temp_state.s7 = state.s7 + k2.s7 * h * 0.5;
        k3 = system_equations(temp_state, R);
        
        // Вычисление k4
        temp_state.s0 = state.s0 + k3.s0 * h;
        temp_state.s1 = state.s1 + k3.s1 * h;
        temp_state.s2 = state.s2 + k3.s2 * h;
        temp_state.s3 = state.s3 + k3.s3 * h;
        temp_state.s4 = state.s4 + k3.s4 * h;
        temp_state.s5 = state.s5 + k3.s5 * h;
        temp_state.s6 = state.s6 + k3.s6 * h;
        temp_state.s7 = state.s7 + k3.s7 * h;
        k4 = system_equations(temp_state, R);
        
        // Обновление состояния
        state.s0 += (k1.s0 + 2.0 * k2.s0 + 2.0 * k3.s0 + k4.s0) * h / 6.0;
        state.s1 += (k1.s1 + 2.0 * k2.s1 + 2.0 * k3.s1 + k4.s1) * h / 6.0;
        state.s2 += (k1.s2 + 2.0 * k2.s2 + 2.0 * k3.s2 + k4.s2) * h / 6.0;
        state.s3 += (k1.s3 + 2.0 * k2.s3 + 2.0 * k3.s3 + k4.s3) * h / 6.0;
        state.s4 += (k1.s4 + 2.0 * k2.s4 + 2.0 * k3.s4 + k4.s4) * h / 6.0;
        state.s5 += (k1.s5 + 2.0 * k2.s5 + 2.0 * k3.s5 + k4.s5) * h / 6.0;
        state.s6 += (k1.s6 + 2.0 * k2.s6 + 2.0 * k3.s6 + k4.s6) * h / 6.0;
        state.s7 += (k1.s7 + 2.0 * k2.s7 + 2.0 * k3.s7 + k4.s7) * h / 6.0;
        
        // Нормализация углов
        double2 angles = normalize_angles(state.s2, state.s3);
        state.s2 = angles.x;
        state.s3 = angles.y;
        
        // Сохранение состояния
        valid_steps++;
        if (valid_steps % save_step == 0 && point_index < max_points) {
            trajectories[id * max_points + point_index] = state;
            point_index++;
        }
        
        // Проверка физических пределов
        if (fabs(state.s0) > 1e6 || fabs(state.s1) > 1e6 || 
            fabs(state.s4) > 1e6 || fabs(state.s5) > 1e6 || 
            fabs(state.s6) > 1e6 || fabs(state.s7) > 1e6) {
            break;
        }
    }
    
    point_counts[id] = point_index;
}