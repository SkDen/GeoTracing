#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
#ifndef M_PI_2
#define M_PI_2 1.5707963267948966192313216916398
#endif

// Безопасное вычисление обратного значения с защитой от деления на ноль
inline double safe_divide(double numerator, double denominator, double epsilon) {
    return denominator != 0.0 ? numerator / (denominator + copysign(epsilon, denominator)) : 0.0;
}

// Функция для безопасного вычисления sin(theta) и cos(theta) с защитой от сингулярностей
void safe_sin_cos(double theta, double* sin_theta, double* cos_theta, double epsilon) {
    *sin_theta = sin(theta);
    *cos_theta = cos(theta);
    
    // Регуляризация вблизи полюсов
    if (fabs(*sin_theta) < epsilon) {
        *sin_theta = copysign(epsilon, *sin_theta);
        
        // Вблизи полюсов используем приближения
        if (theta < M_PI_2 || theta > 3.0 * M_PI_2) {
            *cos_theta = 1.0 - 0.5 * theta * theta;  // Приближение для малых theta
        } else {
            *cos_theta = -1.0 + 0.5 * (theta - M_PI) * (theta - M_PI);  // Приближение для theta near pi
        }
    }
}

// Функция для обработки сингулярных точек
double8 handle_singularities(double8 state, double epsilon) {
    // Проверяем, не слишком ли близко к полюсам
    if (fabs(state.s2) < epsilon || fabs(state.s2 - M_PI) < epsilon) {
        // Если слишком близко к полюсам, обнуляем угловые импульсы
        state.s6 = 0.0;
        state.s7 = 0.0;
    }
    
    return state;
}

// Функция вычисления производных системы с улучшенной обработкой сингулярностей
double8 system_equations(double8 y, double r_s) {
    // Обрабатываем сингулярные точки перед вычислением
    y = handle_singularities(y, 1e-8);
    
    double r = y.s1;
    double theta = y.s2;
    double pr = y.s5;
    double ptheta = y.s6;
    double pphi = y.s7;
    
    // Малая константа для регуляризации сингулярностей
    const double epsilon = 1e-8;
    
    // Безопасное вычисление sin и cos
    double sin_theta, cos_theta;
    safe_sin_cos(theta, &sin_theta, &cos_theta, epsilon);
    
    double sin2_theta = sin_theta * sin_theta;
    double sin3_theta = sin2_theta * sin_theta;
    
    double r2 = r * r;
    double r3 = r2 * r;
    
    double gamma = 1.0 - r_s / r;
    double gamma_diff = r_s / r2;
    
    // Безопасное вычисление gamma_inv_diff с защитой от деления на ноль
    double gamma_inv_diff = 0.0;
    if (fabs(r - r_s) > epsilon) {
        gamma_inv_diff = -r_s / ((r - r_s) * (r - r_s));
    }
    
    double8 dy_dlambda;
    
    // Координаты
    dy_dlambda.s0 = safe_divide(-y.s4, gamma, epsilon);
    dy_dlambda.s1 = gamma * pr;
    dy_dlambda.s2 = safe_divide(ptheta, r2, epsilon);
    dy_dlambda.s3 = safe_divide(pphi, r2 * sin2_theta, epsilon);
    
    // Импульсы
    dy_dlambda.s4 = 0.0;
    
    // Безопасное вычисление для dr/dlambda
    if (fabs(r) > epsilon) {
        dy_dlambda.s5 = gamma_inv_diff * y.s4 * y.s4 * 0.5
                        - gamma_diff * pr * pr * 0.5
                        + safe_divide(ptheta * ptheta, r3, epsilon)
                        + safe_divide(pphi * pphi, r3 * sin2_theta, epsilon);
    } else {
        dy_dlambda.s5 = 0.0;
    }
    
    // Безопасное вычисление для dtheta/dlambda
    if (fabs(sin_theta) > epsilon && fabs(r) > epsilon) {
        dy_dlambda.s6 = safe_divide(pphi * pphi * cos_theta, r2 * sin3_theta, epsilon);
    } else {
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

__kernel void runge_kutta4_tracing(
    const double r_s,
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
    const double epsilon = 1e-8;
    const double epsilon_horizon = 1.1;
    
    // Временные переменные
    double8 k1, k2, k3, k4;
    double8 temp_state;
    
    // Обрабатываем начальное состояние на сингулярные точки
    state = handle_singularities(state, epsilon);
    
    for (int i = 0; i < max_iterations; i++) {
        // Проверяем валидность состояния
        if (!is_valid_state(state)) {
            point_status[id] = state;
            horizon_flag[id] = 3;  // Флаг невалидного состояния
            return;
        }
        
        // Вычисляем коэффициенты Рунге-Кутты
        k1 = system_equations(state, r_s);
        
        temp_state = state + k1 * h2;
        temp_state = handle_singularities(temp_state, epsilon);
        k2 = system_equations(temp_state, r_s);
        
        temp_state = state + k2 * h2;
        temp_state = handle_singularities(temp_state, epsilon);
        k3 = system_equations(temp_state, r_s);
        
        temp_state = state + k3 * h;
        temp_state = handle_singularities(temp_state, epsilon);
        k4 = system_equations(temp_state, r_s);
        
        // Обновляем состояние
        state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * h6;
        
        // Обрабатываем сингулярные точки после обновления
        state = handle_singularities(state, epsilon);
        
        // Определяем пересечение небесной сферы
        if (state.s1 >= R_sky_sphere) {
            point_status[id] = state;
            horizon_flag[id] = 0;
            return;
        }

        // Проверяем выход за горизонт событий
        if (state.s1 < epsilon_horizon * r_s) {
            point_status[id] = state;
            horizon_flag[id] = 1;
            return;
        }
    }

    // В случае превышения количества итераций
    // Фотон движется по орбите ЧД
    point_status[id] = state;
    horizon_flag[id] = 2;
    return;
}