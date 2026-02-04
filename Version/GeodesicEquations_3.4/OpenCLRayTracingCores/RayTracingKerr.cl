#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// System equations for Kerr-Newman metric
double8 system_equations(double8 y, double M, double a, double Q) {
    // Extract variables
    double r = y.s1;
    double theta = y.s2;
    double pr = y.s5;
    double ptheta = y.s6;
    double pphi = y.s7;
    
    // Precalculations
    double r2 = r * r;
    double a2 = a * a;
    double Q2 = Q * Q;
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    double sin2_theta = sin_theta * sin_theta;
    double cos2_theta = cos_theta * cos_theta;
    
    // Metric components
    double Delta = r2 - 2.0*M*r + a2 + Q2;
    double rho2 = r2 + a2 * cos2_theta;
    
    // Avoid division by zero
    double safe_sin2_theta = fmax(sin2_theta, 1e-16);
    double safe_rho2 = fmax(rho2, 1e-16);
    double safe_Delta = fabs(Delta) < 1e-12 ? copysign(1e-12, Delta) : Delta;
    
    // Inverse metric components
    double r2_a2 = r2 + a2;
    double Sigma2 = r2_a2*r2_a2 - a2 * Delta * sin2_theta;
    
    double g_tt = -Sigma2 / (safe_rho2 * safe_Delta);
    double g_tphi = -a * (2.0*M*r - Q2) / (safe_rho2 * safe_Delta);
    double g_phiphi = (Delta - a2 * sin2_theta) / (safe_rho2 * safe_Delta * safe_sin2_theta);
    double g_rr = Delta / safe_rho2;
    double g_thetatheta = 1.0 / safe_rho2;
    
    // Derivatives
    double dDelta_dr = 2.0*r - 2.0*M;
    double drho2_dr = 2.0*r;
    double drho2_dtheta = -2.0*a2*sin_theta*cos_theta;
    
    double dSigma2_dr = 4.0*r*(r2 + a2) - a2 * dDelta_dr * sin2_theta;
    double dSigma2_dtheta = -a2 * Delta * 2.0*sin_theta*cos_theta;
    
    // Metric derivatives
    double denom_tt = safe_rho2 * safe_Delta;
    double dg_tt_dr = -(dSigma2_dr*denom_tt - Sigma2*(drho2_dr*safe_Delta + safe_rho2*dDelta_dr)) / (denom_tt*denom_tt);
    double dg_tt_dtheta = -(dSigma2_dtheta*denom_tt - Sigma2*drho2_dtheta*safe_Delta) / (denom_tt*denom_tt);
    
    double dg_tphi_dr = -a*(2.0*M*denom_tt - (2.0*M*r - Q2)*(drho2_dr*safe_Delta + safe_rho2*dDelta_dr)) / (denom_tt*denom_tt);
    double dg_tphi_dtheta = a*(2.0*M*r - Q2)*drho2_dtheta*safe_Delta / (denom_tt*denom_tt);
    
    double denom_rr = safe_rho2;
    double dg_rr_dr = (dDelta_dr*denom_rr - Delta*drho2_dr) / (denom_rr*denom_rr);
    double dg_rr_dtheta = -Delta*drho2_dtheta / (denom_rr*denom_rr);
    
    double dg_thetatheta_dr = -drho2_dr / (denom_rr*denom_rr);
    double dg_thetatheta_dtheta = -drho2_dtheta / (denom_rr*denom_rr);
    
    double A_phi = Delta - a2*sin2_theta;
    double B_phi = safe_rho2 * safe_Delta * safe_sin2_theta;
    double dA_phi_dr = dDelta_dr;
    double dA_phi_dtheta = -a2*2.0*sin_theta*cos_theta;
    double dB_phi_dr = drho2_dr*safe_Delta*safe_sin2_theta + safe_rho2*dDelta_dr*safe_sin2_theta;
    double dB_phi_dtheta = drho2_dtheta*safe_Delta*safe_sin2_theta + safe_rho2*safe_Delta*2.0*sin_theta*cos_theta;
    
    double dg_phiphi_dr = (dA_phi_dr*B_phi - A_phi*dB_phi_dr) / (B_phi*B_phi);
    double dg_phiphi_dtheta = (dA_phi_dtheta*B_phi - A_phi*dB_phi_dtheta) / (B_phi*B_phi);
    
    // Equations of motion
    double8 dy_dlambda;
    
    // Coordinate equations
    dy_dlambda.s0 = g_tt * y.s4 + g_tphi * pphi;  // dt/dlambda
    dy_dlambda.s1 = g_rr * pr;                    // dr/dlambda
    dy_dlambda.s2 = g_thetatheta * ptheta;        // dtheta/dlambda
    dy_dlambda.s3 = g_tphi * y.s4 + g_phiphi * pphi; // dphi/dlambda
    
    // Momentum equations
    dy_dlambda.s4 = 0.0;  // dp_t/dlambda
    dy_dlambda.s5 = -0.5 * (dg_tt_dr*y.s4*y.s4 + 2.0*dg_tphi_dr*y.s4*pphi + dg_rr_dr*pr*pr + 
                           dg_thetatheta_dr*ptheta*ptheta + dg_phiphi_dr*pphi*pphi);
    dy_dlambda.s6 = -0.5 * (dg_tt_dtheta*y.s4*y.s4 + 2.0*dg_tphi_dtheta*y.s4*pphi + dg_rr_dtheta*pr*pr + 
                           dg_thetatheta_dtheta*ptheta*ptheta + dg_phiphi_dtheta*pphi*pphi);
    dy_dlambda.s7 = 0.0;  // dp_phi/dlambda
    
    return dy_dlambda;
}

// Adaptive step for throat region
double adaptive_step(double r, double pr, double M, double a, double Q, double base_h) {
    double r_plus = M + sqrt(M*M - a*a - Q*Q);
    double distance = fabs(r - r_plus);
    double speed = fabs(pr);
    
    // Minimal step near throat
    double min_step = base_h * 0.01;
    
    // Reduce step near horizon with high speed
    if (distance < 2.0 * r_plus && speed > 0.1) {
        double factor = fmax(0.01, distance / (2.0 * r_plus));
        return base_h * factor;
    }
    
    // Minimal step very close to horizon
    if (distance < 0.5 * r_plus) {
        return min_step;
    }
    
    return base_h;
}

// Check if state is valid
bool is_valid_state(double8 state) {
    return !(isnan(state.s0) || isnan(state.s1) || isnan(state.s2) || isnan(state.s3) ||
             isnan(state.s4) || isnan(state.s5) || isnan(state.s6) || isnan(state.s7) ||
             isinf(state.s0) || isinf(state.s1) || isinf(state.s2) || isinf(state.s3) ||
             isinf(state.s4) || isinf(state.s5) || isinf(state.s6) || isinf(state.s7));
}

__kernel void runge_kutta4_tracing(
    const double M,
    const double a,
    const double Q,
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
    
    // Precomputed constants
    const double h6 = 1.0 / 6.0;
    
    // Temporary variables
    double8 k1, k2, k3, k4;
    double8 temp_state;
    
    // Calculate horizon radius
    double r_plus = M + sqrt(M*M - a*a - Q*Q);
    
    for (int i = 0; i < max_iterations; i++) {
        // Check if state is valid
        if (!is_valid_state(state)) {
            final_states[id] = state;
            horizon_flags[id] = 3;  // Invalid state
            return;
        }
        
        // Adaptive step near horizon
        current_h = adaptive_step(state.s1, state.s5, M, a, Q, h);
        const double h2 = current_h * 0.5;
        
        // Runge-Kutta coefficients
        k1 = system_equations(state, M, a, Q);
        
        temp_state = state + k1 * h2;
        k2 = system_equations(temp_state, M, a, Q);
        
        temp_state = state + k2 * h2;
        k3 = system_equations(temp_state, M, a, Q);
        
        temp_state = state + k3 * current_h;
        k4 = system_equations(temp_state, M, a, Q);
        
        // Update solution
        state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (current_h * h6);
        
        // Check if reached sky sphere
        if (state.s1 >= R_sky_sphere) {
            final_states[id] = state;
            horizon_flags[id] = 0;  // Reached sky sphere
            return;
        }
        
        // Check if crossed horizon (with small margin)
        if (state.s1 < 1.01 * r_plus) {
            final_states[id] = state;
            horizon_flags[id] = 1;  // Crossed horizon
            return;
        }
    }
    
    // Maximum iterations reached
    final_states[id] = state;
    horizon_flags[id] = 2;  // Max iterations
}