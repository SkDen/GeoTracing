
# %%memit
# %load_ext memory_profiler
# -*- coding: utf-8 -*-

import numpy as np

from memory_profiler    import profile

from enums              import MetricType, VectorType, CoordinatesType
from vector4            import Vector4
from metric             import Metric
from camera             import Camera
from OpenClLoader       import OpenClKernelLoader, OpenClKernelLoaderTracing
from visualizer         import TrajectoryVisualizer, TrajectoryAnimator
from ImageConstruction  import PanoramicSkySphereRenderer, InterpolationMethod


def simple_progress_callback(current_frame, total_frames):
    """
    Простая функция обратного вызова для отображения прогресса в консоли
    
    Args:
        current_frame: текущий обрабатываемый кадр
        total_frames: общее количество кадров
    """
    progress = (current_frame + 1) / total_frames * 100
    print(f"Обработано кадров: {current_frame + 1}/{total_frames} ({progress:.1f}%)", end='\r')
    
    # При достижении 100% переводим строку
    if current_frame + 1 == total_frames:
        print("\nЗавершено!")


if __name__ == "__main__":

    # Параметры интегрирования
    h = 0.01            # Шаг интегрирования
    lambda_0 = 0        # Начало интегрирования
    lambda_end = 100     # Конец интегрирования

    # Параметры объектов
    r_s = 2           # Радиус Шварцшильда

    r_0 = 1.5           # Горловина кротовой норы
    m = 20.0
    L_wormhole = 1.0

    M = 1.0             # Масса ЧД Керра
    Q = 0.0             # Заряд ЧД Керра
    L = (M)**2            # Момент вращения ЧД Керра

    goedel_moment = 0.01

    scale_factor=10.0
    k=0.1

    static_scale_factor = 10.0
    hubble_parameter = 0.0
    
    str_kerr = f"""
    Условие максимально возможной эргосферы при нулевом электрическом заряде
    Q = 0                   (заряд объекта)
    M = [пользовательская]  (необходимая масса)
    L = M^2                 (общий момент вращения)
    a = L/M                 (удельный момент вращения)
    """

    # Парметры камеры
    W: int = 200
    H: int = 200

    total_ray: int = W*H

    # Инициализируем метрики
    metric_mincovski = Metric(MetricType.MINKOWSKI)
    metric_ellis = Metric(MetricType.ELLIS_BRONNIKIVA, r_0=r_0)
    metric_schwarzschild = Metric(MetricType.SCHWARZSCHILD, r_s=r_s)
    metric_kerr = Metric(MetricType.KERR_NEWMAN, M=M, Q=Q, L=L)
    metric_goedel = Metric(MetricType.GOEDEL, godel_moment=goedel_moment)
    metric_spherical_universe = Metric(MetricType.SPHERICAL_UNIVERSE, static_scale_factor=static_scale_factor, hubble_parameter=hubble_parameter)
    metric_cylindrical_universe = Metric(MetricType.CYLINDRICAL_UNIVERSE, scale_factor=scale_factor)
    metric_parameter_wormhole = Metric(MetricType.PARAMETERIZED_WORMHOLE, r_0=r_0, L=L_wormhole, m=m)

    current_metric = metric_schwarzschild

    # Инициализируем графики
    # plot_trajectory = TrajectoryVisualizer()
    # plot_trajectory.set_parameters(r_s=r_s, r_0=r_0, 
    #                                M=M, Q=Q, L=L, 
    #                                scale_factor=scale_factor, 
    #                                static_scale_factor=static_scale_factor,
    #                                hubble_parameter=hubble_parameter)
    
    # # Устанавливаем положение камеры
    # plot_trajectory.set_camera_position(elevation=0, azimuth=90, distance=5)
    
    # animator = TrajectoryAnimator()
    # animator.set_parameters(r_0=r_0, r_s=r_s, 
    #                         M=M, Q=Q, L=L, 
    #                         scale_factor=scale_factor, 
    #                         static_scale_factor=static_scale_factor,
    #                         hubble_parameter=hubble_parameter)
    # animator.set_camera_position(elevation=90, azimuth=0, distance=5)

    # Инициализируем вычислительные ядра
    tracer = OpenClKernelLoader(platform_idx=0, device_idx=0)
    ray_tracer = OpenClKernelLoaderTracing(platform_idx=0, device_idx=0)

    # Инициализируем камеру (задаем параметры размера, фокуса, тим используемой метрики)
    camera = Camera(width=W, height=H, focus=0.5, aspect_ratio_inv=False, metric=current_metric)
    # 4-вектор позиции камеры в сферический координатах
    camera_pos = Vector4(t=0, x=15, y=np.pi/2, z=0, vtype=VectorType.COORDINATES, dtype=np.float64)
    # Задаем поворот камеры
    camera.set_direction(theta=0, phi=0)
    # Инициализируем массив начальных импульсов для фотонов камеры
    initial_states_cov = camera.create_camera_rays(camera_position=camera_pos, energy=1, dtype=np.float64)

    # # Проверяем ковариантные 4-импульсы на коректность перехода из локального базиса в глобальный
    # #   Просто определяем норму вектора импульса путем скалярного произведения с метрическим тензором 
    # vect_cov = initial_states_cov[..., 0, 4:]
    # print(current_metric.scalar_product_cov_cov(camera_pos, vect_cov, vect_cov))

    # step = 10
    # initial_states_cov = np.zeros((step, 8))
    # values = np.linspace(0, 10, step)
    # for i, val in enumerate(values):
    #     initial_states_cov[i] = np.array([0, val, 0, 0, -7, 1, 1, 1])


    point, flag = ray_tracer.ray_tracing(MetricType.SCHWARZSCHILD,
                                        initial_states_cov,
                                        max_iterations=100000,
                                        h=h,
                                        R_sky_sphere=100,
                                        Width=W,
                                        Height=H,
                                        r_0=r_0,
                                        r_s=r_s,
                                        Q=Q,
                                        M=M,
                                        L=L,
                                        a=L_wormhole/2,
                                        m=m,
                                        dtype=np.float64,
                                        save_path="file_array\\full_array",
                                        return_results=True
                                        )
    

    renderer_lanczos = PanoramicSkySphereRenderer(
        point, flag, 
        panorama_path="SkySpheres\\metro_noord_2k.png",
        panorama_another_path="SkySpheres\\limpopo_golf_course_2k.png",
        interpolation_method=InterpolationMethod.LANCZOS
    )

    # Построение изображений
    image_lanczos = renderer_lanczos.render_with_gradient_compensation(gradient_strength=0.01)

    # Сохранение лучшего результата
    renderer_lanczos.save_image(image_lanczos, "Image.png")

    # np.savetxt('GeodesicEquations\\full_array.txt', point[50], delimiter='\t', encoding='utf-8', fmt='%.3f')
    # np.savetxt('GeodesicEquations\\flags.txt', flag[50], delimiter='\t', encoding='utf-8', fmt='%.3f')





    # if (total_ray <= 1000):
    #     trajectories, point_counts = tracer.compute_trajectories(
    #         MetricType.SPHERICAL_UNIVERSE,
    #         initial_states_cov,
    #         lambda_0,
    #         lambda_end,
    #         h,
    #         W*H,
    #         max_points=int((lambda_end - lambda_0) / h),
    #         save_step=1,
    #         static_scale_factor=static_scale_factor,
    #         hubble_parameter=hubble_parameter,
    #         r_s=r_s,
    #         r_0=r_0,
    #         M=M,
    #         Q=Q,
    #         L=L,
    #         dtype=np.float64)

    #     # Визуализация
    #     plot_trajectory.plot(trajectories, point_counts, total_ray,
    #                         coordinates_type=CoordinatesType.HYPERSPHERIC,
    #                         type_surfaces=MetricType.SPHERICAL_UNIVERSE,
    #                         step_ray=1,
    #                         axis_scaling_factor=0.5,
    #                         avto_mashtab=False,
    #                         black_hole=True,
    #                         grids_of_spaces=True
    #                         )







    # Создание анимации из одного набора траекторий
    # animator.create_animation(
    #     trajectories=trajectories,
    #     point_counts=point_counts,
    #     total_ray=W*H,
    #     output_filename="trajectories_animation.mp4",
    #     fps=100,
    #     coordinates_type=CoordinatesType.SPHERICAL,
    #     type_surfaces=MetricType.SCHWARZSCHILD,
    #     step_ray=1,
    #     axis_scaling_factor=0.5,
    #     avto_mashtab=False,
    #     black_hole=True,
    #     grids_of_spaces=False,
    #     dpi=100,
    #     progress_callback=simple_progress_callback
    # )

    # # Создание анимации с вращением камеры вокруг оси Y
    # animator.create_animation_with_rotation(
    #     trajectories=trajectories,
    #     point_counts=point_counts,
    #     total_ray=W*H,
    #     output_filename="rotation_animation.mp4",
    #     fps=60,
    #     dpi=100,
    #     coordinates_type=CoordinatesType.SPHERICAL,
    #     type_surfaces=MetricType.SCHWARZSCHILD,
    #     rotation_speed=0.5,  # градусов за кадр
    #     rotation_axis='y',   # вращение вокруг оси Y
    #     initial_elev=30,     # начальный угол возвышения
    #     initial_azim=-60,     # начальный азимутальный угол
    #     progress_callback=simple_progress_callback
    # )

# %%
