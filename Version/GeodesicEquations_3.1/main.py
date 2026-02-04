
# %%memit
# %load_ext memory_profiler
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from memory_profiler import profile

from enums import MetricType, VectorType, CoordinatesType
from vector4 import Vector4
from metric import Metric
from camera import Camera
from OpenClLoader import OpenClKernelLoader
from visualizer import TrajectoryVisualizer


if __name__ == "__main__":

    # Параметры интегрирования
    h = 0.01            # Шаг интегрирования
    lambda_0 = 0        # Начало интегрирования
    lambda_end = 100     # Конец интегрирования

    # Параметры обьектов
    r_s = 3.0           # Радиус Шварцшильда
    r_0 = 3.0           # Горловина кротовой норы

    M = 5.0             # Масса ЧД Керра
    Q = 1.0             # Заряд ЧД Керра
    L = 24.0            # Момент вращения ЧД Керра

    goedel_moment = 10.0

    scale_factor=10.0
    k=0.1

    static_scale_factor = 20.0
    hubble_parameter = 0.0
    
    str_kerr = f"""
    Условие максимально возможной эргосферы при нулевом электрическом заряде
    Q = 0                   (заряд объекта)
    M = [пользовательская]  (необходимая масса)
    L = M^2                 (общий момент вращения)
    a = L/M                 (удельный момент вращения)
    """

    # Парметры камеры
    W: int = 10
    H: int = 10

    # Инициализируем метрики
    metric_mincovski = Metric(MetricType.MINKOWSKI)
    metric_ellis = Metric(MetricType.ELLIS_BRONNIKIVA, r_0=r_0)
    metric_schwarzschild = Metric(MetricType.SCHWARZSCHILD, r_s=r_s)
    metric_kerr = Metric(MetricType.KERR_NEWMAN, M=M, Q=Q, L=L)
    metric_goedel = Metric(MetricType.GOEDEL, godel_moment=goedel_moment)
    metric_spherical_universe = Metric(MetricType.SPHERICAL_UNIVERSE, 
                                       static_scale_factor=static_scale_factor, hubble_parameter=hubble_parameter)
    metric_cylindrical_universe = Metric(MetricType.CYLINDRICAL_UNIVERSE, scale_factor=scale_factor)

    current_metric = metric_spherical_universe

    # Инициализируем графики
    plot_trajectory = TrajectoryVisualizer()
    plot_trajectory.set_parameters(r_0=r_0, r_s=r_s, 
                                   M=M, Q=Q, L=L, 
                                   scale_factor=scale_factor, 
                                   static_scale_factor=static_scale_factor,
                                   hubble_parameter=hubble_parameter)

    # Инициализируем вычислительное ядро
    tracer = OpenClKernelLoader(platform_idx=0, device_idx=0)

    # Инициализируем камеру (задаем параметры размера, фокуса, тим используемой метрики)
    camera = Camera(width=W, height=H, focus=2, aspect_ratio_inv=False, metric=current_metric)
    # 4-вектор позиции камеры в сферический координатах
    camera_pos = Vector4(t=0, x=np.pi/6, y=np.pi/2, z=0, vtype=VectorType.COORDINATES, dtype=np.float64)
    # Задаем поворот камеры
    camera.set_direction(theta=0, phi=0)
    # Инициализируем массив начальных импульсов для фотонов камеры
    initial_states_cov = camera.create_camera_rays(camera_position=camera_pos, energy=1, dtype=np.float64)

    # Проверяем ковариантные 4-импульсы на коректность перехода из локального базиса в глобальный
    #   Просто определяем норму вектора импульса путем скалярного произведения с метрическим тензором 
    # vect_cov = initial_states_cov[..., 0, 4:]
    # print(current_metric.scalar_product_cov_cov(camera_pos, vect_cov, vect_cov))

    # step = 10
    # initial_states_cov = np.zeros((step, 8))
    # values = np.linspace(0, 10, step)
    # for i, val in enumerate(values):
    #     initial_states_cov[i] = np.array([0, val, np.pi/6, 0, -7, 1, 0, 10])

    trajectories, point_counts = tracer.compute_trajectories(
        MetricType.SPHERICAL_UNIVERSE,
        initial_states_cov,
        lambda_0,
        lambda_end,
        h,
        W*H,
        max_points=int((lambda_end - lambda_0) / h),
        save_step=1,
        static_scale_factor=static_scale_factor,
        hubble_parameter=hubble_parameter,
        dtype=np.float64)

    np.savetxt('GeodesicEquations\\full_array.txt', trajectories[0], delimiter='\t', encoding='utf-8', fmt='%.3f')

    # Визуализация
    plot_trajectory.plot(trajectories, point_counts, W, H,
                         CoordinatesType.HYPERSPHERIC,
                         MetricType.SPHERICAL_UNIVERSE,
                         step_ray=1,
                         axis_scaling_factor=1.0,
                         avto_mashtab=False,
                         black_hole=False,
                         grids_of_spaces=True
                         )    


# %%
