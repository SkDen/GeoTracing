from typing import Tuple
import numpy as np
import os

from ImageConstruction import PanoramicSkySphereRenderer, InterpolationMethod


def file_chek(point_status: np.ndarray):
    # Проверяем массив данных
    file = point_status.reshape(-1, 8)
    np.savetxt('GeodesicEquations\\full_array.txt', file, delimiter='\t', encoding='utf-8', fmt='%.3f')

if __name__ == "__main__":
    # file_rays = "file_array_best\\WormHole.npz"
    # file_rays = "file_array_best\\see_horizont.npz"
    # file_rays = "file_array_best\\mouth_of_wormhole.npz"
    file_rays = "file_array_best\\wormHoleSee.npz"
    # file_rays = "file_array\\full_array_final.npz"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, file_rays)
    data = np.load(path)

    # Извлечение массивов по именам
    point_status_2d = data['point_status']
    point_flag_2d = data['point_flag']

    # Закрываем файл
    data.close()
    file_chek(point_status_2d)

    renderer_lanczos = PanoramicSkySphereRenderer(
        point_status_2d, point_flag_2d, 
        panorama_path="GeodesicEquations\\SkySpheres\\metro_noord_2k.png",
        panorama_another_path="GeodesicEquations\\SkySpheres\\limpopo_golf_course_2k.png",
        interpolation_method=InterpolationMethod.LANCZOS
    )

    # Построение изображений с различными методами
    image_lanczos = renderer_lanczos.render_with_gradient_compensation(gradient_strength=0.2, 
                                                                       rotation_sphere_phi=-np.pi/2)
    renderer_lanczos.save_image(image_lanczos, "Image.png")

    # mass_angle = np.linspace(0, 2*np.pi, 72)
    # for i, angle in enumerate(mass_angle):
    #     image_lanczos = renderer_lanczos.render_with_gradient_compensation(gradient_strength=0.2, 
    #                                                                        rotation_sphere_phi=-np.pi/2)
    #     renderer_lanczos.save_image(image_lanczos, f"ImageRotation\\Image_{i}.png")
    

