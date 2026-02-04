
# %%memit
# %load_ext memory_profiler
# -*- coding: utf-8 -*-

import os
import sys

import pyopencl as cl
import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from memory_profiler import profile

from enums import MetricType, VectorType
from config import file_mapping, PYOPENCL_CTX

from vector4 import Vector4
from metric import Metric
from camera import Camera

# Настройка окружения
os.environ['PYOPENCL_CTX'] = PYOPENCL_CTX

    
def plot_trajectories(trajectories, point_counts, W, H,
                      type_plot=MetricType.SCHWARZSCHILD,
                      r_s=1.0,
                      r_0=1.0,
                      M=1.0,
                      Q=1.0,
                      L=1.0,
                      step_ray=1, avto_mashtab=True, black_hole=True):
    """
    Строит 3D график траекторий геодезических
    """

    # Функция для построения поверхности
    def Graph(ax, r, color, alpha, num, name):
        phi = np.linspace(0, 2 * np.pi, num)
        theta = np.linspace(0, np.pi, num)
        phi, theta = np.meshgrid(phi, theta)

        # проверяем явлется ли параметр функцией или переменной
        if callable(r):
            R = r(theta, phi)
        else:
            R = r

        X = R * np.sin(theta) * np.cos(phi)
        Y = R * np.sin(theta) * np.sin(phi)
        Z = R * np.cos(theta)

        surf = ax.plot_surface(X, Y, Z, color=color, alpha=alpha, label=name)
        return surf

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Преобразование сферических координат в декартовы
    def to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    # Счетчик построенных траекторий
    trajectory_count = 0
    
    # Цикл по всем траекториям с заданным шагом
    for idx in range(0, W*H, step_ray):
        num_points = point_counts[idx]
        if num_points < 2:
            continue
            
        # Извлекаем координаты
        traj = trajectories[idx][:num_points]
        r = traj[:, 1]  # Радиальная координата
        theta = traj[:, 2]  # Угол theta (полярный угол)
        phi = traj[:, 3]  # Угол phi (азимутальный угол)
        
        # Фильтрация точек за горизонтом
        valid = r > 1.001 * r_s
        r = r[valid]
        theta = theta[valid]
        phi = phi[valid]
        
        if len(r) < 2:
            continue
            
        # Преобразование координат
        x, y, z = to_cartesian(r, theta, phi)
        
        # Вычисление цвета на основе индекса траектории
        color_value = idx / (W * H)
        color = plt.cm.viridis(color_value)
        
        # Построение траектории
        ax.plot(x, y, z, 
                linewidth=0.7, 
                alpha=0.8,
                color=color)
        
        trajectory_count += 1

    print(f"Построено траекторий: {trajectory_count}")

    if black_hole == True:
        if type_plot == MetricType.SCHWARZSCHILD:
            # Визуализация черной дыры Шварцшильда
            Graph(ax, r_0, 'black', 0.7, 50, 'Горизонт событий')
            ax.set_title('Траектории света в метрике Шварцшильда\n(3D декартовы координаты)', 
                         fontsize=14)

            # Фотонная сфера (круг в экваториальной плоскости)
            phi = np.linspace(0, 2*np.pi, 100)
            r_photon = 1.5 * r_s
            x_ph = r_photon * np.cos(phi)
            y_ph = r_photon * np.sin(phi)
            z_ph = np.zeros_like(phi)
            ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7, label='Фотонная сфера')
            
            # Фотонная сфера (круг вокруг оси y)
            x_ph = r_photon * np.cos(phi)
            z_ph = r_photon * np.sin(phi)
            y_ph = np.zeros_like(phi)
            ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7)

            # Фотонная сфера (круг вокруг оси x)
            z_ph = r_photon * np.cos(phi)
            y_ph = r_photon * np.sin(phi)
            x_ph = np.zeros_like(phi)
            ax.plot(x_ph, y_ph, z_ph, 'r-', lw=2, alpha=0.7)

        elif type_plot == MetricType.ELLIS_BRONNIKIVA:
            # Визуализация червоточины Эллиса-Бронникова
            Graph(ax, r_0, 'black', 0.7, 50, 'Горловина червоточины')
            ax.set_title('Траектории света в метрике Эллиса-Бронникова\n(3D декартовы координаты)', 
                         fontsize=14)

        elif type_plot == MetricType.KERR_NEWMAN:
            # Визуализируем горизонты в метрике Керра-Ньюмена

            # Радиус эргосфер
            # Внешная эргосфера
            def r_erg_pl(theta, phi):
                param = M**2 - Q**2 - (L/M)**2 * np.cos(theta)**2
                return np.where(param >= 0, M + np.sqrt(param), np.nan)
            # Внутренная эргосфера
            def r_erg_in(theta, phi):
                param = M**2 - Q**2 - (L/M)**2 * np.cos(theta)**2
                return np.where(param >= 0, M - np.sqrt(param), np.nan)
            
            # Радиусы горизонтов
            # Внешний горизонт
            def r_pl(theta, phi):
                param = M**2 - Q**2 - (L/M)**2
                r_val = M + np.sqrt(param) if param >= 0 else np.nan
                return np.full_like(theta, r_val)
            # Внутренний горизонт
            def r_in(theta, phi):
                param = M**2 - Q**2 - (L/M)**2
                r_val = M - np.sqrt(param) if param >= 0 else np.nan
                return np.full_like(theta, r_val)
            
            color_erg_pl = (0.0, 1.0, 0.0, 0.4)
            color_erg_in = (0.53, 0.0, 0.87, 0.4)
            color_pl = (1.0, 0.0, 0.0, 0.4)
            color_in = (1.0, 0.0, 1.0, 0.4)

            # Неотрисовываем эргосферы если вращение отсутствует
            if not L == 0:
                Graph(ax, r_erg_pl, color_erg_pl, None, 30, 'Внешняя эргосфера')
                Graph(ax, r_erg_in, color_erg_in, None, 10, 'Внутренняя эргосфера')

            Graph(ax, r_pl, color_pl, None, 30, 'Внешний горизонт событий')
            Graph(ax, r_in, color_in, None, 10, 'Внутренний горизонт событий')

            if L > 0:
                ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах.\n'+
                            f'M={M}, L={L}, Q={Q}. Вращение в направлении увелечения координаты φ.\n'+
                            f'[вокруг оси Z по часовой]', 
                            fontsize=12)
            elif L < 0:
                ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах\n'+
                            f'M={M}, L={L}, Q={Q}. Вращение в направлении уменьшения координаты φ.\n'+
                            f'[вокруг оси Z против часовой]', 
                            fontsize=12)
            else:
                ax.set_title(f'Траектории света в метрике Керра-Ньюмена. Проекция в 3D декартовых координатах\n'+
                            f'M={M}, L={L}, Q={Q}. Вращение отсутствует.\n'+
                            f'[вокруг оси Z против часовой]', 
                            fontsize=12)


    # Настройки визуализации
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Автомасштабирование
    all_points = trajectories[:, :, 1:4].reshape(-1, 3)
    valid_points = all_points[all_points[:, 0] > 1.001 * r_s]
    
    if avto_mashtab == True:
        if valid_points.size > 0:
            max_val = np.max(valid_points)
            min_val = np.min(valid_points)
            max_range = max(np.abs(max_val), np.abs(min_val)) * 1.1
            
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
    else:
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_zlim(-18, 18)

    
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def OpenCL_platform():
    # Получаем список всех платформ
    platforms = cl.get_platforms()

    print("\nДоступные платформы и устройства:")
    for i, platform in enumerate(platforms):
        print(f"Платформа {i}: {platform.name}")
        print(f"  Вендор: {platform.vendor}")
        print(f"  Версия: {platform.version}")
        
        # Получаем устройства на этой платформе
        devices = platform.get_devices()
        for j, device in enumerate(devices):
            print(f"  Устройство {j}:")
            print(f"    Имя: {device.name}")
            print(f"    Тип: {cl.device_type.to_string(device.type)}")
            print(f"    Макс. размер рабочей группы: {device.max_work_group_size}")
            print(f"    Макс. вычислительные единицы: {device.max_compute_units}")
            print(f"    Глобальная память: {device.global_mem_size/(1024**3):.2f} GB")
            print(f"    Локальная память: {device.local_mem_size/1024:.2f} KB")
            print(f"    Макс. размер буфера: {device.max_mem_alloc_size/(1024**3):.2f} GB")

def OpenCL_using_device(ctx):
    # Выводим информацию о используемых устройств
    print("Устройства в текущем контексте:")
    for i, device in enumerate(ctx.devices):
        print(f"  Устройство {i}: {device.name}")
        print(f"    Тип: {cl.device_type.to_string(device.type)}")
        
        # Получаем информацию о вычислительных возможностях
        if hasattr(device, 'max_work_item_sizes'):
            print(f"    Макс. размер рабочего элемента: {device.max_work_item_sizes}")
        # # Проверяем расширения
        # if hasattr(device, 'extensions'):
        #     print(f"    Поддерживаемые расширения: {device.extensions}")

# @profile
def OpenCL_NEW(metric_type, initial_states, lambda_0, lambda_end, h, global_size, 
           max_points=1000, save_step=10, r_s=1.0, 
           r_0=1.0, 
           M=3.0, Q=1.0, L=1.0,
           dtype=np.float64):
    
    # Определяем название файла ядра
    kernel = file_mapping[metric_type]
    
    # Определяем число итераций в ядре для заданного времени моделирования
    iterations = int((lambda_end - lambda_0) / h)
    
    # Определяем максимальное число лучей
    width, height = global_size
    total_rays = width * height

    # Преобразуем двухмерный массив импульсов (i, j) в одномерный
    initial_states_flat = initial_states.reshape(-1, 8)

    # Инициализация контекста OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    # Выводим информацию о максимальной памяти
    device = ctx.devices[0]
    max_alloc_size = device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    global_mem_size = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
    print(f"\nМакс. размер буфера: {max_alloc_size/(1024**2):.2f} MB")
    print(f"Общая память: {global_mem_size/(1024**2):.2f} MB")

    # Загрузка кода ядра и компиляция программы
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(script_dir, kernel)
    with open(kernel_path, encoding='utf-8') as f:
        kernel_code = f.read()
    prg = cl.Program(ctx, kernel_code).build()

    # Выделение памяти для результатов
    trajectories = np.zeros((total_rays, max_points, 8), dtype=dtype)
    point_counts = np.zeros(total_rays, dtype=np.int32)

    # Размер батча
    batch_size = min(total_rays, 2500)  # Безопасный размер батча
    batches = (total_rays + batch_size - 1) // batch_size

    # Вычисляем максимальные размеры буферов
    max_initial_size = batch_size * 8 * np.dtype(dtype).itemsize
    max_trajectories_size = batch_size * max_points * 8 * np.dtype(dtype).itemsize
    max_points_size = batch_size * np.dtype(np.int32).itemsize

    # Проверяем ограничения памяти устройства
    if max_trajectories_size > max_alloc_size:
        # Автоматическая коррекция размера батча
        batch_size = max_alloc_size // (max_points * 8 * np.dtype(dtype).itemsize)
        max_initial_size = batch_size * 8 * np.dtype(dtype).itemsize
        max_trajectories_size = batch_size * max_points * 8 * np.dtype(dtype).itemsize
        max_points_size = batch_size * np.dtype(np.int32).itemsize
        batches = (total_rays + batch_size - 1) // batch_size
        print(f"Размер батча уменьшен до {batch_size} из-за ограничений памяти устройства")

    # Создаем буферы максимального размера
    mf = cl.mem_flags
    
    # Выводим информацию о размерах буферов
    print(f"\nРазмер буфера начальных условий: {max_initial_size/(1024**2):.2f} МБ")
    print(f"Размер буфера траекторий: {max_trajectories_size/(1024**2):.2f} МБ")
    print(f"Размер буфера счетчиков: {max_points_size/(1024**2):.2f} МБ")

    # Выводим некоторую информацию
    print(f"\nКоличество моделируемых лучей: {total_rays}")
    print(f"Общее число итераций в ядре: {iterations}")
    print(f"Временной интервал моделирования: {h*iterations} сек")
    print(f"Максимальное количество батчей: {batches}")

    # Расчитываем размер выходного массива с траекториями
    total_mb = sys.getsizeof(trajectories) / (1024 * 1024)
    print(f"Размер выходного массива траекторий: {total_mb:.2f} МБ\n")

    start_total = time.time()
    for batch_idx in range(batches):
        start_batch = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_rays)
        current_batch_size = end_idx - start_idx

        # Создаем буферы для текущего батча
        initial_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=initial_states_flat[start_idx:end_idx])
        trajectories_buf = cl.Buffer(ctx, mf.WRITE_ONLY, max_trajectories_size)
        point_counts_buf = cl.Buffer(ctx, mf.WRITE_ONLY, max_points_size)

        if kernel == file_mapping[MetricType.SCHWARZSCHILD]:
            # Запуск ядра для текущего батча
            prg.runge_kutta4_trajectories(
                queue, 
                (current_batch_size,),  # 1D NDRange
                None,
                dtype(r_s),
                dtype(h),
                np.int32(iterations),
                np.int32(max_points),
                np.int32(save_step),
                initial_buf,
                trajectories_buf,
                point_counts_buf
            )
        elif kernel == file_mapping[MetricType.ELLIS_BRONNIKIVA]:
            # Запуск ядра для текущего батча
            prg.runge_kutta4_trajectories(
                queue, 
                (current_batch_size,),
                None,
                dtype(r_0),
                dtype(h),
                np.int32(iterations),
                np.int32(max_points),
                np.int32(save_step),
                initial_buf,
                trajectories_buf,
                point_counts_buf
            )
        elif kernel == file_mapping[MetricType.KERR_NEWMAN]:
            # Запуск ядра для текущего батча
            prg.runge_kutta4_trajectories(
                queue, 
                (current_batch_size,),
                None,
                dtype(M),
                dtype(L/M),
                dtype(Q),
                dtype(h),
                np.int32(iterations),
                np.int32(max_points),
                np.int32(save_step),
                initial_buf,
                trajectories_buf,
                point_counts_buf
            )

        # Чтение результатов батча
        trajectories_batch_flat = np.empty(current_batch_size * max_points * 8, dtype=dtype)
        point_counts_batch = np.empty(current_batch_size, dtype=np.int32)
        
        # Копирование данных с устройства
        cl.enqueue_copy(
            queue, 
            trajectories_batch_flat, 
            trajectories_buf
        )
        cl.enqueue_copy(
            queue, 
            point_counts_batch, 
            point_counts_buf
        )
        
        # Ожидаем завершения операций
        queue.finish()
        
        # Проверяем размеры перед reshape
        expected_size = current_batch_size * max_points * 8
        if trajectories_batch_flat.size != expected_size:
            raise ValueError(
                f"Несоответствие размеров: ожидалось {expected_size}, "
                f"получено {trajectories_batch_flat.size}"
            )
        
        # Сохранение результатов
        trajectories_batch = trajectories_batch_flat.reshape(current_batch_size, max_points, 8)
        trajectories[start_idx:end_idx] = trajectories_batch
        point_counts[start_idx:end_idx] = point_counts_batch
        
        # Освобождаем буферы
        initial_buf.release()
        trajectories_buf.release()
        point_counts_buf.release()
        
        print(f"Батч {batch_idx+1}/{batches} ({current_batch_size} лучей) занял: {time.time() - start_batch:.2f} сек")
    print(f"Всего вычислено {total_rays} траекторий за {time.time() - start_total:.2f} секунд\n")
    
    return trajectories, point_counts

# @profile
def OpenCL(metric_type, initial_states, lambda_0, lambda_end, h, global_size, 
           max_points=1000, save_step=10, r_s=1.0, 
           r_0=1.0, 
           M=3.0, Q=1.0, L=1.0,
           dtype=np.float64):
    
    # Определыем название файла ядра
    kernel = file_mapping[metric_type]
    
    # Определяем число итераций в ядре для заданного времени моделирования
    iterations = int((lambda_end - lambda_0) / h)
    
    # Определяем максимальное число лучей
    width, height = global_size
    total_rays = width * height

    # Преобразуем двухмерный массив импульсов (i, j) в одномерный
    initial_states_flat = initial_states.reshape(-1, 8)

    # Инициализация контекста OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    # OpenCL_using_device(ctx)

    # Выводим информацию о максимальной памяти
    device = ctx.devices[0]
    max_alloc_size = device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    global_mem_size = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
    print(f"\nМакс. размер буфера: {max_alloc_size/(1024**2):.2f} MB")
    print(f"Общая память: {global_mem_size/(1024**2):.2f} MB")

    # Загрузка кода ядра и компиляция программы
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(script_dir, kernel)
    with open(kernel_path, encoding='utf-8') as f:
        kernel_code = f.read()
    prg = cl.Program(ctx, kernel_code).build()

    # Выделение памяти для результатов
    trajectories = np.zeros((total_rays, max_points, 8), dtype=dtype)
    point_counts = np.zeros(total_rays, dtype=np.int32)

    # Размер батча
    batch_size = min(total_rays, 5000)  # Безопасный размер батча
    batches = (total_rays + batch_size - 1) // batch_size

    # Вычисляем максимальные размеры буферов
    max_initial_size = batch_size * 8 * np.dtype(dtype).itemsize
    max_trajectories_size = batch_size * max_points * 8 * np.dtype(dtype).itemsize
    max_points_size = batch_size * np.dtype(np.int32).itemsize

    # Проверяем ограничения памяти устройства
    if max_trajectories_size > max_alloc_size:
        # Автоматическая коррекция размера батча
        max_batch_size = max_alloc_size // (max_points * 8 * np.dtype(dtype).itemsize)
        max_initial_size = max_batch_size * 8 * np.dtype(dtype).itemsize
        max_trajectories_size = max_batch_size * max_points * 8 * np.dtype(dtype).itemsize
        max_points_size = max_batch_size * np.dtype(np.int32).itemsize
        print(f"Размер батча уменьшен до {max_batch_size} из-за ограничений памяти устройства")

    # Создаем буферы максимального размера
    mf = cl.mem_flags
    initial_buf = cl.Buffer(ctx, mf.READ_WRITE, max_initial_size)
    trajectories_buf = cl.Buffer(ctx, mf.WRITE_ONLY, max_trajectories_size)
    point_counts_buf = cl.Buffer(ctx, mf.WRITE_ONLY, max_points_size)

    # Выводим информацию о размерах буферов
    print(f"\nРазмер буфера начальных условий: {max_initial_size/(1024**2):.2f} МБ")
    print(f"Размер буфера траекторий: {max_trajectories_size/(1024**2):.2f} МБ")
    print(f"Размер буфера счетчиков: {max_points_size/(1024**2):.2f} МБ")

    # Выводим некоторую информацию
    print(f"\nКоличество моделируемых лучей: {total_rays}")
    print(f"Общее число итераций в ядре: {iterations}")
    print(f"Временной интервал моделирования: {h*iterations} сек")
    print(f"Максимальное количество батчей: {batches}")

    # Расчитываем размер выходного массива с траекториями
    total_mb = sys.getsizeof(trajectories) / (1024 * 1024)
    print(f"Размер выходного массива траекторий: {total_mb:.2f} МБ\n")

    start_total = time.time()
    for batch_idx in range(batches):
        start_batch = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_rays)
        current_batch_size = end_idx - start_idx

        # Копируем начальные условия для текущего батча
        initial_batch = initial_states_flat[start_idx:end_idx]
        # Правильное копирование данных на устройство
        cl.enqueue_copy(
            queue, 
            initial_buf, 
            initial_batch
        )

        if kernel == file_mapping[MetricType.SCHWARZSCHILD]:
            # Запуск ядра для текущего батча
            prg.runge_kutta4_trajectories(
                queue, 
                (current_batch_size,),  # 1D NDRange
                None,
                dtype(r_s),
                dtype(h),
                np.int32(iterations),
                np.int32(max_points),
                np.int32(save_step),
                initial_buf,
                trajectories_buf,
                point_counts_buf
            )
        elif kernel == file_mapping[MetricType.ELLIS_BRONNIKIVA]:
            # Запуск ядра для текущего батча
            prg.runge_kutta4_trajectories(
                queue, 
                (current_batch_size,),
                None,
                dtype(r_0),
                dtype(h),
                np.int32(iterations),
                np.int32(max_points),
                np.int32(save_step),
                initial_buf,
                trajectories_buf,
                point_counts_buf
            )
        elif kernel == file_mapping[MetricType.KERR_NEWMAN]:
            # Запуск ядра для текущего батча
            prg.runge_kutta4_trajectories(
                queue, 
                (current_batch_size,),
                None,
                dtype(M),
                dtype(L/M),
                dtype(Q),
                dtype(h),
                np.int32(iterations),
                np.int32(max_points),
                np.int32(save_step),
                initial_buf,
                trajectories_buf,
                point_counts_buf
            )

        # Чтение результатов батча
        trajectories_batch_flat = np.empty(current_batch_size * max_points * 8, dtype=dtype)
        point_counts_batch = np.empty(current_batch_size, dtype=np.int32)
        
        # Копирование данных с устройства
        cl.enqueue_copy(
            queue, 
            trajectories_batch_flat, 
            trajectories_buf
        )
        cl.enqueue_copy(
            queue, 
            point_counts_batch, 
            point_counts_buf
        )
        
        # Ожидаем завершения операций
        queue.finish()
        
        # Проверяем размеры перед reshape
        expected_size = current_batch_size * max_points * 8
        if trajectories_batch_flat.size != expected_size:
            raise ValueError(
                f"Несоответствие размеров: ожидалось {expected_size}, "
                f"получено {trajectories_batch_flat.size}"
            )
        
        # Сохранение результатов
        trajectories_batch = trajectories_batch_flat.reshape(current_batch_size, max_points, 8)
        trajectories[start_idx:end_idx] = trajectories_batch
        point_counts[start_idx:end_idx] = point_counts_batch
        
        print(f"Батч {batch_idx+1}/{batches} ({current_batch_size} лучей) занял: {time.time() - start_batch:.2f} сек")
    print(f"Всего вычислено {total_rays} траекторий за {time.time() - start_total:.2f} секунд\n")
    
    # Освобождаем ресурсы
    initial_buf.release()
    trajectories_buf.release()
    point_counts_buf.release()

    return trajectories, point_counts


if __name__ == "__main__":
    # Вывод информации о устройствах пралельного вычисления
    # OpenCL_platform()

    # Параметры интегрирования
    h = 0.01            # Шаг интегрирования
    lambda_0 = 0        # Начало интегрирования
    lambda_end = 50     # Конец интегрирования

    # Параметры обьектов
    r_s = 3.0           # Радиус Шварцшильда
    r_0 = 3.0           # Горловина кротовой норы
    M = 2.0             # Масса ЧД Керра
    Q = 0.0             # Заряд ЧД Керра
    L = M*M             # Момент вращения ЧД Керра


    str_kerr = f"""
    Условие максимально возможной эргосферы при нулевом электрическом заряде
    Q = 0                   (заряд объекта)
    M = [пользовательская]  (необходимая масса)
    L = M^2                 (общий момент вращения)
    a = L/M                 (удельный момент вращения)
    """

    # Парметры камеры
    W = 30
    H = 30

    # Инициализируем метрики
    metric_mincovski = Metric(MetricType.MINKOWSKI)
    metric_ellis = Metric(MetricType.ELLIS_BRONNIKIVA, r_0=r_0)
    metric_schwarzschild = Metric(MetricType.SCHWARZSCHILD, r_s=r_s)
    metric_kerr = Metric(MetricType.KERR_NEWMAN, M=M, Q=Q, L=L)

    # Инициализируем камеру (задаем параметры размера, фокуса, тим используемой метрики)
    camera = Camera(width=W, height=H, focus=2, aspect_ratio_inv=False, metric=metric_kerr)

    # 4-вектор позиции камеры в сферический координатах
    camera_pos = Vector4(t=0, x=20, y=np.pi/2, z=0, vtype=VectorType.COORDINATES, dtype=np.float64)
    
    # Задаем поворот камеры
    camera.set_direction(theta=0, phi=0)
    
    # Инициализируем массив начальных импульсов для фотонов камеры
    initial_states_cov = camera.create_camera_rays(camera_position=camera_pos, energy=1, dtype=np.float64)

    # Проверяем ковариантные 4-импульсы на коректность перехода из локального базиса в глобальный
    #   Просто определяем норму вектора импульса путем скалярного произведения с метрическим тензором 
    # vect_cov = initial_states_cov[..., 0, 4:]
    # print(metric_kerr.scalar_product_cov_cov(camera_pos, vect_cov, vect_cov))

    # posihion = Vector4(t=0, x=10, y=0, z=0, vtype=VectorType.COORDINATES, dtype=np.float64)
    # vect_impulse_local_cont = Vector4(x=1, y=0, z=0, vtype=VectorType.IMPULSE_PHOTON_COV, dtype=np.float64).to_array()
    # vect_impulse_local_cont *= 10
    # vect_impulse_glob_kerr_cov = metric_kerr.local_to_global_vector_cont_cont(posihion, vect_impulse_local_cont)
    # print(metric_kerr.scalar_product_contra_contra(None, vect_impulse_glob_kerr_cov, vect_impulse_glob_kerr_cov))

    # step = 10
    # init_mass = np.zeros((step, 8))
    # values = np.linspace(0, 2*np.pi, step)
    # for i, val in enumerate(values):
    #     init_mass[i] = np.array([0, 30, np.pi/2, val, -10, 0, 0, 0])

    # Вычисляем траектории
    trajectories, point_counts = OpenCL(
        MetricType.KERR_NEWMAN,
        initial_states_cov,
        lambda_0,
        lambda_end,
        h,
        (W, H),
        max_points=int((lambda_end - lambda_0) / h),
        save_step=1,
        r_s=r_s,
        r_0=r_0,
        M=M,
        Q=Q,
        L=L,
        dtype = np.float64
    )

    np.savetxt('GeodesicEquations\\full_array.txt', trajectories[0], delimiter='\t', encoding='utf-8', fmt='%.3f')

    # Визуализация
    plot_trajectories(trajectories, point_counts, W, H,
                      type_plot=MetricType.KERR_NEWMAN,
                      r_s=r_s,
                      r_0=r_0, 
                      M=M,
                      Q=Q,
                      L=L,
                      step_ray=2, avto_mashtab=False, black_hole=True)

# %%
