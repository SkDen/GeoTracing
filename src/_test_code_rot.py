import numpy as np
import time


from enums import MetricType, VectorType
from vector4 import Vector4
from metric import Metric
from camera import Camera
from OpenClLoader import OpenClKernelLoader, OpenClKernelLoaderTracing
from ImageConstruction import PanoramicSkySphereRenderer, InterpolationMethod



if __name__ == "__main__":

    # Параметры интегрирования
    h = 0.03
    max_iter = 100000
    R_sky_sphere = 40
    
    # Параметры червоточины
    r_0: float = 2.0
    m: float = 3.0
    L: float = 1.0

    # Парметры камеры
    W: int = 400
    H: int = 400
    total_ray: int = W*H

    # Инициализируем метрики
    metric_ellis = Metric(MetricType.ELLIS_BRONNIKIVA, r_0=r_0)
    metric_parameter_wormhole = Metric(MetricType.PARAMETERIZED_WORMHOLE, r_0=r_0, L=L, m=m)
    current_metric = metric_ellis

    # Инициализируем ядро
    ray_tracer = OpenClKernelLoaderTracing(platform_idx=0, device_idx=0)

    cadr = 200
    mass_r = np.linspace(-5, 5, cadr)
    mass_phi = np.linspace(0, 2*np.pi, cadr)
    mass_r_phi = np.zeros((cadr, 2))
    
    mass_r_phi[:, 0] = mass_r[:]
    mass_r_phi[:, 1] = mass_phi[:]

    for i, val in enumerate(mass_r_phi):
        # Контроль времени
        # time_s = time.time()

        # Строим системму импульсов фотонов камеры
        camera = Camera(width=W, height=H, focus=0.7, aspect_ratio_inv=False, metric=current_metric)
        camera_pos = Vector4(t=0, x=val[0], y=np.pi/2, z=val[1], vtype=VectorType.COORDINATES, dtype=np.float64)
        camera.set_direction(theta=0, phi=5*np.pi/6)
        initial_states_cov = camera.create_camera_rays(camera_position=camera_pos, energy=1, dtype=np.float64)

        point, flag = ray_tracer.ray_tracing(MetricType.ELLIS_BRONNIKIVA,
                                            initial_states_cov,
                                            max_iterations=max_iter,
                                            h=h,
                                            R_sky_sphere=R_sky_sphere,
                                            Width=W,
                                            Height=H,
                                            r_0=r_0,
                                            a=L/2,
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
        image_lanczos = renderer_lanczos.render_with_gradient_compensation(gradient_strength=0.2)
        # Сохранение лучшего результата
        renderer_lanczos.save_image(image_lanczos, f"ImagePassingRotation\\Image_{i}.png")

        # print(f"Рендер: {time.time() - time_s:.2f}")