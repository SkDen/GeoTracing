import cv2
import os
import glob
import re
import numpy as np
from PIL import Image

def extract_number(filename):
    """
    Извлекает число из имени файла для сортировки
    """
    match = re.search(r'_(\d+)\.png$', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def create_video_from_multiple_sources(sources_sequence, output_video_path, fps=30):
    """
    Создает видео из изображений, расположенных в разных директориях, 
    с соблюдением заданной последовательности
    
    Args:
        sources_sequence (list): Список словарей с описанием источников и их порядка
            Пример: [
                {"pattern": "path/to/Image_*.png", "count": 5},
                {"pattern": "path/to/Image_Passing_*.png", "count": 3},
                {"pattern": "path/to/Image_*.png", "count": 2}
            ]
        output_video_path (str): Путь для сохранения видео файла
        fps (int): Количество кадров в секунду
    """
    
    # Собираем все кадры в правильном порядке
    all_frames = []
    
    for source in sources_sequence:
        pattern = source["pattern"]
        count = source.get("count", None)  # Если None, используем все изображения из этого источника
        
        # Получаем файлы изображений для этого источника
        image_files = glob.glob(pattern)
        
        # Сортируем файлы по номеру в названии
        image_files.sort(key=extract_number)
        
        if not image_files:
            print(f"Предупреждение: не найдены файлы по шаблону {pattern}")
            continue
        
        # Ограничиваем количество, если указано
        if count is not None:
            image_files = image_files[:count]
        
        # Добавляем в общий список
        all_frames.extend(image_files)
        print(f"Добавлено {len(image_files)} кадров из {pattern}")
    
    if not all_frames:
        print("Не найдены файлы изображений по указанным шаблонам")
        return
    
    # Определяем размер кадра из первого изображения
    first_image = cv2.imread(all_frames[0])
    if first_image is None:
        print(f"Не удалось загрузить первое изображение: {all_frames[0]}")
        return
    
    height, width, layers = first_image.shape
    frame_size = (width, height)
    
    # Определяем кодек и создаем объект VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    print(f"Создание видео из {len(all_frames)} изображений...")
    
    # Добавляем кадры в видео
    for i, image_path in enumerate(all_frames):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            continue
            
        # Изменяем размер, если нужно
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, frame_size)
            
        out.write(img)
        
        # Выводим прогресс каждые 10 кадров
        if i % 10 == 0:
            print(f"Обработано {i+1}/{len(all_frames)}: {os.path.basename(image_path)}")
    
    # Закрываем VideoWriter
    out.release()
    print(f"Видео сохранено как: {output_video_path}")

def create_video_with_dynamic_sequence(source_patterns, sequence, output_video_path, fps=30):
    """
    Создает видео с динамической последовательностью кадров из разных источников
    
    Args:
        source_patterns (dict): Словарь с шаблонами источников
            Пример: {
                "main": "path/to/Image_*.png",
                "passing": "path/to/Image_Passing_*.png"
            }
        sequence (list): Последовательность использования источников
            Пример: ["main", "main", "passing", "main", "passing"]
        output_video_path (str): Путь для сохранения видео файла
        fps (int): Количество кадров в секунду
    """
    
    # Загружаем изображения из всех источников
    sources = {}
    for name, pattern in source_patterns.items():
        image_files = glob.glob(pattern)
        image_files.sort(key=extract_number)
        sources[name] = image_files
        print(f"Загружено {len(image_files)} изображений из источника '{name}'")
    
    # Создаем последовательность кадров
    all_frames = []
    frame_counters = {name: 0 for name in source_patterns.keys()}  # Счетчики для каждого источника
    
    for source_name in sequence:
        if source_name not in sources:
            print(f"Предупреждение: источник '{source_name}' не найден")
            continue
        
        source_images = sources[source_name]
        if not source_images:
            print(f"Предупреждение: источник '{source_name}' не содержит изображений")
            continue
        
        # Добавляем следующий кадр из этого источника
        if frame_counters[source_name] < len(source_images):
            all_frames.append(source_images[frame_counters[source_name]])
            frame_counters[source_name] += 1
        else:
            # Если изображения в источнике закончились, используем последнее
            all_frames.append(source_images[-1])
            print(f"Предупреждение: в источнике '{source_name}' закончились изображения, используется последнее")
    
    if not all_frames:
        print("Не удалось создать последовательность кадров")
        return
    
    # Определяем размер кадра из первого изображения
    first_image = cv2.imread(all_frames[0])
    height, width, layers = first_image.shape
    frame_size = (width, height)
    
    # Создаем видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    print(f"Создание видео из {len(all_frames)} изображений...")
    
    # Добавляем кадры в видео
    for i, image_path in enumerate(all_frames):
        img = cv2.imread(image_path)
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, frame_size)
        out.write(img)
    
    # Закрываем VideoWriter
    out.release()
    print(f"Видео сохранено как: {output_video_path}")

# Примеры использования
if __name__ == "__main__":
    # Пример 1: Простая последовательность источников
    sources_sequence = [
        {"pattern": "RotationAroundWormhole/Image_*.png", "count": 200},
        {"pattern": "ImagePassing/Image_Passing_*.png", "count": 200},
        {"pattern": "RotationCamera/Image_*.png", "count": 100}
    ]
    
    create_video_from_multiple_sources(sources_sequence, "Render_video.mp4", fps=19)
    
    # # Пример 2: Динамическая последовательность
    # source_patterns = {
    #     "main": "ImageRotation/Image_*.png",
    #     "passing": "PassingRotation/Image_Passing_*.png"
    # }
    
    # # Создаем сложную последовательность
    # sequence = ["main"] * 3 + ["passing"] * 2 + ["main"] * 2 + ["passing"] * 1 + ["main"] * 4
    
    # create_video_with_dynamic_sequence(source_patterns, sequence, "dynamic_video.mp4", fps=24)
    
    # Пример 3: Создание видео с переходом между сценами
    # Сначала все кадры из первого источника, затем все из второго
    # all_main = glob.glob("ImageRotation/Image_*.png")
    # all_main.sort(key=extract_number)
    
    # all_passing = glob.glob("PassingRotation/Image_Passing_*.png")
    # all_passing.sort(key=extract_number)
    
    # # Создаем переход: последние 5 кадров из первого источника и первые 5 из второго
    # # с плавным смешиванием
    # transition_frames = 10
    # transition_sequence = []
    
    # # Добавляем основные кадры (кроме последних 5)
    # transition_sequence.extend([{"pattern": f"ImageRotation/Image_{i}.png", "count": 1} 
    #                            for i in range(len(all_main) - 5)])
    
    # # Добавляем переходные кадры
    # for i in range(transition_frames):
    #     alpha = i / transition_frames
    #     # Для каждого кадра перехода создаем свой шаблон
    #     # (в реальности нужно будет создать эти кадры заранее)
    #     transition_sequence.append({
    #         "pattern": f"Transitions/Transition_{i:03d}.png", 
    #         "count": 1
    #     })
    
    # # Добавляем кадры из второго источника
    # transition_sequence.append({"pattern": "PassingRotation/Image_Passing_*.png", "count": None})
    
    # create_video_from_multiple_sources(transition_sequence, "transition_video.mp4", fps=24)