import os
import sys

import pyopencl as cl
import numpy as np
import time

from typing import Dict, Tuple, Optional

from enums import MetricType
from config import file_mapping, file_ray_tracing


class OpenClKernelLoader:
    def __init__(self, platform_idx: int = 0, device_idx: int = 0):
        """
        Инициализация OpenCl_Kernel_Loader
        
        Аргументы:
            platform_idx: индекс платформы OpenCL
            device_idx: индекс устройства на платформе
        """
        self.platforms = cl.get_platforms()
        self.platform = self.platforms[platform_idx]
        self.devices = self.platform.get_devices()
        self.device = self.devices[device_idx]
        
        # Создание контекста и очереди команд
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        
        # Кэш для скомпилированных программ
        self.programs = {}
        
        # Информация о памяти устройства
        self.max_alloc_size = self.device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
        self.global_mem_size = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
        
        print(f"Инициализирован OpenCL Ray Tracer на устройстве: {self.device.name}")
    
    def print_platform_info(self):
        """Вывод информации о доступных платформах и устройствах"""
        print("\nДоступные платформы и устройства:")
        for i, platform in enumerate(self.platforms):
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
    
    def print_used_device_info(self):
        """Вывод информации о используемом устройстве"""
        print("Используемое устройство:")
        print(f"  Имя: {self.device.name}")
        print(f"  Тип: {cl.device_type.to_string(self.device.type)}")
        print(f"  Макс. размер рабочей группы: {self.device.max_work_group_size}")
        print(f"  Макс. вычислительные единицы: {self.device.max_compute_units}")
        print(f"  Глобальная память: {self.global_mem_size/(1024**3):.2f} GB")
        print(f"  Макс. размер буфера: {self.max_alloc_size/(1024**3):.2f} GB")
        
        # Дополнительная информация, если доступна
        if hasattr(self.device, 'max_work_item_sizes'):
            print(f"  Макс. размер рабочего элемента: {self.device.max_work_item_sizes}")
    
    def _compile_program(self, metric_type: MetricType) -> cl.Program:
        """Компиляция программы OpenCL для заданного типа метрики"""
        if metric_type in self.programs:
            return self.programs[metric_type]
        
        # Определяем дирректорию файла ядра
        kernel = file_mapping[metric_type]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(script_dir, kernel)
        
        # Читаем файл ядра
        with open(kernel_path, encoding='utf-8') as f:
            kernel_code = f.read()
        
        # Компилируем файл ядра и хешируем его
        program = cl.Program(self.ctx, kernel_code).build()
        self.programs[metric_type] = program
        return program
    
    def compute_trajectories(self, 
                             metric_type: MetricType,
                             initial_states: np.ndarray,
                             lambda_0: float,
                             lambda_end: float,
                             h: float,
                             total_rays: int,
                             max_points: int = 1000,
                             save_step: int = 10,
                             r_s: float = 1.0,
                             r_0: float = 1.0, a: float = 1.0, m: float = 1.0,
                             M: float = 3.0, Q: float = 1.0, L: float = 1.0,
                             goedel_moment: float = 1.0,
                             scale_factor: float = 1.0, k: float = 1.0,
                             static_scale_factor: float = 1.0, hubble_parameter: float = 1.0,
                             dtype: type = np.float64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление траекторий лучей в заданной метрике
        
        Args:
            metric_type: тип метрики пространства-времени
            initial_states: начальные состояния лучей
            lambda_0: начальное значение аффинного параметра
            lambda_end: конечное значение аффинного параметра
            h: шаг интегрирования
            global_size: размер глобальной рабочей области (width, height)
            max_points: максимальное количество точек на траекторию
            save_step: шаг сохранения точек
            ...: параметры метрики
            
        Returns:
            trajectories: массив траекторий размером (total_rays, max_points, 8)
            point_counts: количество точек для каждого луча
        """
        # Компилируем программу для выбранной метрики
        program = self._compile_program(metric_type)
        
        # Определяем число итераций
        iterations = int((lambda_end - lambda_0) / h)
        
        # Преобразуем начальные состояния в одномерный массив
        initial_states_flat = initial_states.reshape(-1, 8)
        
        # Выделение памяти для результатов
        trajectories = np.zeros((total_rays, max_points, 8), dtype=dtype)
        point_counts = np.zeros(total_rays, dtype=np.int32)
        
        # Определяем размер батча с учетом ограничений памяти
        batch_size = self._calculate_batch_size_max_points(total_rays, max_points, dtype)
        batches = (total_rays + batch_size - 1) // batch_size
        
        # Создаем буферы
        mf = cl.mem_flags
        initial_buf = cl.Buffer(self.ctx, mf.READ_WRITE, batch_size * 8 * np.dtype(dtype).itemsize)
        trajectories_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, batch_size * max_points * 8 * np.dtype(dtype).itemsize)
        point_counts_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, batch_size * np.dtype(np.int32).itemsize)
        
        # Выводим информацию о вычислениях
        self._print_computation_info(total_rays, iterations, h, batches, batch_size, trajectories)
        
        # Основной цикл по батчам
        start_total = time.time()
        for batch_idx in range(batches):
            start_batch = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_rays)
            current_batch_size = end_idx - start_idx
            
            # Копируем начальные условия для текущего батча
            initial_batch = initial_states_flat[start_idx:end_idx]
            cl.enqueue_copy(self.queue, initial_buf, initial_batch)
            
            # Запускаем ядро в зависимости от типа метрики
            self._run_kernel(program, 
                             metric_type, 
                             current_batch_size, 
                             initial_buf, 
                             trajectories_buf, 
                             point_counts_buf, 
                             h, 
                             iterations, 
                             max_points, 
                             save_step,
                             r_s, a, m,
                             r_0, 
                             M, Q, L,
                             goedel_moment,
                             scale_factor, k, 
                             static_scale_factor, hubble_parameter, 
                             dtype)
            
            # Чтение результатов
            trajectories_batch_flat = np.empty(current_batch_size * max_points * 8, dtype=dtype)
            point_counts_batch = np.empty(current_batch_size, dtype=np.int32)
            
            cl.enqueue_copy(self.queue, trajectories_batch_flat, trajectories_buf)
            cl.enqueue_copy(self.queue, point_counts_batch, point_counts_buf)
            
            # Ожидаем завершения операций
            self.queue.finish()
            
            # Проверяем размеры и сохраняем результаты
            expected_size = current_batch_size * max_points * 8
            if trajectories_batch_flat.size != expected_size:
                raise ValueError(
                    f"Несоответствие размеров: ожидалось {expected_size}, "
                    f"получено {trajectories_batch_flat.size}"
                )
            
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
    
    def _calculate_batch_size_max_points(self, total_rays: int, max_points: int, dtype: type) -> int:
        """Вычисление размера батча с учетом ограничений памяти"""
        item_size = np.dtype(dtype).itemsize
        max_batch_size = self.max_alloc_size // (max_points * 8 * item_size)
        batch_size = min(total_rays, 5000, max_batch_size)
        
        if batch_size < total_rays:
            print(f"Размер батча уменьшен до {batch_size} из-за ограничений памяти устройства")
        
        return batch_size
    
    def _print_computation_info(self, total_rays: int, iterations: int, h: float, 
                              batches: int, batch_size: int, trajectories: np.ndarray):
        """Вывод информации о предстоящих вычислениях"""
        print(f"\nМакс. размер буфера: {self.max_alloc_size/(1024**2):.2f} MB")
        print(f"Общая память: {self.global_mem_size/(1024**2):.2f} MB")
        
        max_initial_size = batch_size * 8 * np.dtype(trajectories.dtype).itemsize
        max_trajectories_size = batch_size * trajectories.shape[1] * 8 * np.dtype(trajectories.dtype).itemsize
        max_points_size = batch_size * np.dtype(np.int32).itemsize
        
        print(f"\nРазмер буфера начальных условий: {max_initial_size/(1024**2):.2f} МБ")
        print(f"Размер буфера траекторий: {max_trajectories_size/(1024**2):.2f} МБ")
        print(f"Размер буфера счетчиков: {max_points_size/(1024**2):.2f} МБ")
        
        print(f"\nКоличество моделируемых лучей: {total_rays}")
        print(f"Общее число итераций в ядре: {iterations}")
        print(f"Временной интервал моделирования: {h*iterations} сек")
        print(f"Максимальное количество батчей: {batches}")
        
        total_mb = sys.getsizeof(trajectories) / (1024 * 1024)
        print(f"Размер выходного массива траекторий: {total_mb:.2f} МБ\n")
    
    def _run_kernel(self, program: cl.Program, metric_type: MetricType, 
                    current_batch_size: int, initial_buf: cl.Buffer, 
                    trajectories_buf: cl.Buffer, point_counts_buf: cl.Buffer,
                    h: float, 
                    iterations: int,
                    max_points: int,
                    save_step: int,
                    r_s: float, a: float, m: float,
                    r_0: float,
                    M: float, Q: float, L: float,
                    goedel_moment: float, scale_factor: float, k: float,
                    static_scale_factor: float, hubble_parameter: float,
                    dtype: type):
        """Запуск соответствующего ядра в зависимости от типа метрики"""

        # Базовые аргументы для меттода Рунге-Кутты в ядре
        # Параметры сохранения точек и передаваемые буферы
        kernel_args = [
            self.queue, 
            (current_batch_size,),
            None,
            dtype(h),
            np.int32(iterations),
            np.int32(max_points),
            np.int32(save_step),
            initial_buf,
            trajectories_buf,
            point_counts_buf
        ]
        
        # Добавляем специфичные для метрики параметры
        if metric_type == MetricType.MINKOWSKI:
            program.runge_kutta4_trajectories(*kernel_args[:3], *kernel_args[3:])
        elif metric_type == MetricType.SCHWARZSCHILD:
            program.runge_kutta4_trajectories(*kernel_args[:3], dtype(r_s), *kernel_args[3:])
        elif metric_type == MetricType.ELLIS_BRONNIKIVA:
            program.runge_kutta4_trajectories(*kernel_args[:3], dtype(r_0), *kernel_args[3:])
        elif metric_type == MetricType.KERR_NEWMAN:
            program.runge_kutta4_trajectories(*kernel_args[:3], dtype(M), dtype(L/M), dtype(Q), *kernel_args[3:])
        elif metric_type == MetricType.GOEDEL:
            program.runge_kutta4_trajectories(*kernel_args[:3], dtype(goedel_moment), *kernel_args[3:])
        elif metric_type == MetricType.FRIEDMAN_ROBERTSON:
            program.runge_kutta4_trajectories(*kernel_args[:3], dtype(scale_factor), dtype(k), *kernel_args[3:])
        elif metric_type == MetricType.SPHERICAL_UNIVERSE:
            program.runge_kutta4_trajectories(*kernel_args[:3], dtype(hubble_parameter), dtype(static_scale_factor), *kernel_args[3:])
        elif metric_type == MetricType.CYLINDRICAL_UNIVERSE:
            program.runge_kutta4_trajectories(*kernel_args[:3], dtype(scale_factor), *kernel_args[3:])
        elif metric_type == MetricType.PARAMETERIZED_WORMHOLE:
            program.runge_kutta4_trajectories(*kernel_args[:3], dtype(r_0), dtype(a), dtype(m), *kernel_args[3:])










class OpenClKernelLoaderTracing(OpenClKernelLoader):
    def __init__(self, platform_idx: int = 0, device_idx: int = 0):
        """
        Инициализация OpenClKernelLoaderTracing для трассировки лучей
        
        Аргументы:
            platform_idx: индекс платформы OpenCL
            device_idx: индекс устройства на платформе
        """
        super().__init__(platform_idx, device_idx)
        
    def _compile_program_ray_tracing(self, metric_type: MetricType) -> cl.Program:
        """Компиляция программы OpenCL для заданного типа метрики"""
        if metric_type in self.programs:
            return self.programs[metric_type]
        
        # Определяем директорию файла ядра
        kernel = file_ray_tracing[metric_type]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(script_dir, kernel)
        
        # Читаем файл ядра
        with open(kernel_path, encoding='utf-8') as f:
            kernel_code = f.read()
        
        # Компилируем файл ядра и кэшируем его
        program = cl.Program(self.ctx, kernel_code).build()
        self.programs[metric_type] = program
        return program

    def _calculate_batch_size(self, total_rays: int, dtype: type) -> int:
        """Вычисление размера батча с учетом ограничений памяти"""
        item_size = np.dtype(dtype).itemsize
        
        # Учитываем размер всех буферов: начальные условия + результаты + флаги
        memory_per_ray = (8 * item_size) + (8 * item_size) + 4
        
        # Оставляем запас памяти для системных нужд (20%)
        available_memory = self.global_mem_size * 0.8
        max_batch_size = int(available_memory // memory_per_ray)
        
        # Ограничиваем максимальный размер батча
        batch_size = min(total_rays, 2000, max_batch_size)
        
        if batch_size < total_rays:
            print(f"Размер батча уменьшен до {batch_size} из-за ограничений памяти устройства")
            print(f"Доступно памяти: {available_memory/(1024**3):.2f} GB")
            print(f"Требуется на батч: {batch_size * memory_per_ray/(1024**3):.6f} GB")
        
        return batch_size

    def ray_tracing(self, 
                    metric_type: MetricType,
                    initial_states: np.ndarray,
                    max_iterations: float,
                    h: float,
                    R_sky_sphere: float,
                    Width: int,
                    Height: int,
                    r_s: float = 1.0,
                    r_0: float = 1.0, a: float = 1.0, m: float = 1.0,
                    M: float = 3.0, Q: float = 1.0, L: float = 1.0,
                    goedel_moment: float = 1.0,
                    scale_factor: float = 1.0, k: float = 1.0,
                    static_scale_factor: float = 1.0, hubble_parameter: float = 1.0,
                    dtype: type = np.float64,
                    save_path: Optional[str] = None,
                    return_results: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Трассировка лучей в заданной метрике
        
        Args:
            metric_type: тип метрики пространства-времени
            initial_states: начальные состояния лучей
            max_iterations: максимальное количество итераций
            h: шаг интегрирования
            R_sky_sphere: радиус небесной сферы
            Width: ширина изображения
            Height: высота изображения
            save_path: путь для сохранения результатов (опционально)
            return_results: флаг, указывающий нужно ли возвращать результаты в память
            ...: параметры метрики
            
        Returns:
            Если return_results=True: кортеж (point_status, point_flag)
            Если return_results=False: None
        """
        # Компилируем программу для выбранной метрики
        program = self._compile_program_ray_tracing(metric_type)

        # Определяем количество лучей
        total_rays = Width * Height
        
        # Сохраняем оригинальную форму для восстановления после вычислений
        original_shape = initial_states.shape[:-1]  # (Height, Width)
        
        # Преобразуем начальные состояния в одномерный массив
        initial_states_flat = initial_states.reshape(-1, 8)
        
        # Выделение памяти для результатов (только если нужно возвращать результаты)
        if return_results:
            point_status = np.zeros((total_rays, 8), dtype=dtype)
            point_flag = np.zeros(total_rays, dtype=np.int32)
        else:
            point_status = None
            point_flag = None
        
        # Определяем размер батча с учетом ограничений памяти
        batch_size = self._calculate_batch_size(total_rays, dtype)
        batches = (total_rays + batch_size - 1) // batch_size
        
        # Выводим информацию о вычислениях
        if return_results:
            self._print_computation_info(total_rays, max_iterations, h, batches, batch_size, point_status)
        else:
            print(f"\nКоличество моделируемых лучей: {total_rays}")
            print(f"Общее число итераций в ядре: {max_iterations}")
            print(f"Временной интервал моделирования: {h*max_iterations} сек")
            print(f"Максимальное количество батчей: {batches}\n")
        
        # Основной цикл по батчам
        start_total = time.time()
        for batch_idx in range(batches):
            start_batch = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_rays)
            current_batch_size = end_idx - start_idx
            
            # Расчет размеров буферов в байтах
            double_size = np.dtype(dtype).itemsize
            int32_size = np.dtype(np.int32).itemsize
            
            initial_buf_size = current_batch_size * 8 * double_size
            point_status_buf_size = current_batch_size * 8 * double_size
            point_flag_buf_size = current_batch_size * int32_size
            
            # Создаем буферы для текущего батча
            mf = cl.mem_flags
            initial_buf = cl.Buffer(self.ctx, mf.READ_WRITE, initial_buf_size)
            point_status_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, point_status_buf_size)
            point_flag_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, point_flag_buf_size)
            
            try:
                # Копируем начальные условия для текущего батча
                initial_batch = initial_states_flat[start_idx:end_idx]
                cl.enqueue_copy(self.queue, initial_buf, initial_batch)
                
                # Запускаем ядро в зависимости от типа метрики
                self._run_kernel_ray_tracing(program, 
                                            metric_type, 
                                            current_batch_size, 
                                            initial_buf, 
                                            point_status_buf, 
                                            point_flag_buf, 
                                            h,
                                            R_sky_sphere,
                                            max_iterations,
                                            r_s, a, m,
                                            r_0, 
                                            M, Q, L,
                                            goedel_moment,
                                            scale_factor, k, 
                                            static_scale_factor, hubble_parameter, 
                                            dtype)
                
                # Чтение результатов
                trajectories_batch_flat = np.empty(current_batch_size * 8, dtype=dtype)
                point_counts_batch = np.empty(current_batch_size, dtype=np.int32)
                
                cl.enqueue_copy(self.queue, trajectories_batch_flat, point_status_buf)
                cl.enqueue_copy(self.queue, point_counts_batch, point_flag_buf)
                
                # Ожидаем завершения операций
                self.queue.finish()
                
                # Проверяем размеры
                expected_size = current_batch_size * 8
                if trajectories_batch_flat.size != expected_size:
                    raise ValueError(
                        f"Несоответствие размеров: ожидалось {expected_size}, "
                        f"получено {trajectories_batch_flat.size}"
                    )
                
                trajectories_batch = trajectories_batch_flat.reshape(current_batch_size, 8)
                
                # Сохраняем или записываем результаты в зависимости от режима
                if return_results:
                    point_status[start_idx:end_idx] = trajectories_batch
                    point_flag[start_idx:end_idx] = point_counts_batch
                
                # Сохранение промежуточных результатов в файл
                if save_path:
                    batch_save_path = f"{save_path}_batch_{batch_idx}.npz"
                    np.savez(batch_save_path,
                             point_status=trajectories_batch,
                             point_flag=point_counts_batch,
                             start_idx=start_idx,
                             batch_size=current_batch_size,
                             width=Width)
                
                print(f"Батч {batch_idx+1}/{batches} ({current_batch_size} лучей) занял: {time.time() - start_batch:.2f} сек")
            
            finally:
                # Гарантированное освобождение ресурсов
                initial_buf.release()
                point_status_buf.release()
                point_flag_buf.release()
        
        print(f"Всего вычислено {total_rays} траекторий за {time.time() - start_total:.2f} секунд\n")

        # Финальное сохранение результатов
        if save_path and return_results:
            # Преобразуем результаты обратно в двухмерную форму
            point_status_2d = point_status.reshape(*original_shape, 8)
            point_flag_2d = point_flag.reshape(original_shape)
            
            final_save_path = f"{save_path}_final.npz"
            np.savez(final_save_path,
                     point_status=point_status_2d,
                     point_flag=point_flag_2d)
            print(f"Финальные результаты сохранены в {final_save_path}")
        
        # Возвращаем результаты или None
        if return_results:
            # Преобразуем результаты обратно в двухмерную форму
            point_status_2d = point_status.reshape(*original_shape, 8)
            point_flag_2d = point_flag.reshape(original_shape)
            return point_status_2d, point_flag_2d
        else:
            return None, None

    def _run_kernel_ray_tracing(self, program: cl.Program, metric_type: MetricType, 
                    current_batch_size: int, initial_buf: cl.Buffer, 
                    point_status_buf: cl.Buffer, point_flag_buf: cl.Buffer,
                    h: float,
                    R_sky_sphere: float,
                    max_iterations: int,
                    r_s: float, a: float, m: float,
                    r_0: float,
                    M: float, Q: float, L: float,
                    goedel_moment: float, scale_factor: float, k: float,
                    static_scale_factor: float, hubble_parameter: float,
                    dtype: type):
        # Получаем ядро
        kernel_name = "runge_kutta4_tracing"
        kernel = getattr(program, kernel_name)
        
        # Формируем аргументы для ядра
        kernel_args = []
        
        # Добавляем специфичные для метрики параметры
        if metric_type == MetricType.SCHWARZSCHILD:
            kernel_args.append(np.float64(r_s))
        elif metric_type == MetricType.ELLIS_BRONNIKIVA:
            kernel_args.append(np.float64(r_0))
        elif metric_type == MetricType.KERR_NEWMAN:
            kernel_args.append(np.float64(M))
            kernel_args.append(np.float64(L/M))
            kernel_args.append(np.float64(Q))
        elif metric_type == MetricType.GOEDEL:
            kernel_args.append(np.float64(goedel_moment))
        elif metric_type == MetricType.FRIEDMAN_ROBERTSON:
            kernel_args.append(np.float64(scale_factor))
            kernel_args.append(np.float64(k))
        elif metric_type == MetricType.SPHERICAL_UNIVERSE:
            kernel_args.append(np.float64(hubble_parameter))
            kernel_args.append(np.float64(static_scale_factor))
        elif metric_type == MetricType.CYLINDRICAL_UNIVERSE:
            kernel_args.append(np.float64(scale_factor))
        elif metric_type == MetricType.PARAMETERIZED_WORMHOLE:
            kernel_args.append(np.float64(r_0))
            kernel_args.append(np.float64(a))
            kernel_args.append(np.float64(m))
        
        # Добавляем общие параметры
        kernel_args.extend([
            np.float64(h),
            np.float64(R_sky_sphere),
            np.int32(max_iterations),
            initial_buf,
            point_status_buf,
            point_flag_buf
        ])
        
        # Устанавливаем аргументы ядра
        kernel.set_args(*kernel_args)
        
        # Определяем размеры рабочей группы
        global_size = (current_batch_size,)
        max_work_group_size = self.device.max_work_group_size
        
        # Вычисляем оптимальный local_size
        if current_batch_size <= max_work_group_size:
            local_size = (current_batch_size,)
        else:
            # Ищем наибольший делитель current_batch_size, который <= max_work_group_size
            local_size_val = max_work_group_size
            while local_size_val > 1:
                if current_batch_size % local_size_val == 0:
                    break
                local_size_val -= 1
            else:
                local_size_val = 1
            local_size = (local_size_val,)
        
        # Запускаем ядро
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)