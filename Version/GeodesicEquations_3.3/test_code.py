import numpy as np
import os

def load_final_results(file_path):
    """
    Загрузка финальных результатов из NPZ-файла
    
    Args:
        file_path: путь к NPZ-файлу с результатами
        
    Returns:
        point_status: массив конечных состояний лучей
        point_flag: флаги состояния для каждого луча
        
    Raises:
        FileNotFoundError: если файл не существует
        KeyError: если в файле нет ожидаемых массивов
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    try:
        with np.load(file_path) as data:
            point_status = data['point_status']
            point_flag = data['point_flag']
        return point_status, point_flag
    except KeyError as e:
        print(f"В файле отсутствуют ожидаемые массивы: {e}")
        print(f"Доступные массивы в файле: {list(data.keys())}")
        raise

# Использование с обработкой ошибок
if __name__ == "__main__":
    try:
        # Попробуйте загрузить финальный файл
        point, flag = load_final_results("GeodesicEquations\\file_array\\full_array_final.npz")
        print("Успешно загружены финальные результаты")
        print(f"Форма point: {point.shape}")
        print(f"Форма flag: {flag.shape}")
        print(point)
    except FileNotFoundError:
        print("Финальный файл не найден. Попробуем найти файлы батчей...")
        
        # Поиск файлов батчей
        import glob
        batch_files = glob.glob("результаты_batch_*.npz")
        
        if batch_files:
            print(f"Найдены файлы батчей: {batch_files}")
            # Загрузка первого батча для примера
            try:
                point, flag = load_final_results(batch_files[0])
                print(f"Загружен батч {batch_files[0]}")
                print(f"Форма point: {point.shape}")
                print(f"Форма flag: {flag.shape}")
            except Exception as e:
                print(f"Ошибка при загрузке батча: {e}")
        else:
            print("Файлы батчей также не найдены.")
            print("Убедитесь, что:")
            print("1. Вы сначала запустили вычисления с сохранением результатов")
            print("2. Указали правильный путь к файлам")
            print("3. Файлы действительно существуют")