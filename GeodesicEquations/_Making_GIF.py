import glob
import os
import re
from PIL import Image
import numpy as np

def extract_number(filename):
    """
    Извлекает число из имени файла для сортировки
    """
    match = re.search(r'_(\d+)\.png$', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def create_gif_from_multiple_sources(sources_sequence, output_gif_path, duration=100, loop=0):
    """
    Создает GIF из изображений, расположенных в разных директориях, 
    с соблюдением заданной последовательности
    
    Args:
        sources_sequence (list): Список словарей с описанием источников и их порядка
            Пример: [
                {"pattern": "path/to/Image_*.png", "count": 5},
                {"pattern": "path/to/Image_Passing_*.png", "count": 3},
                {"pattern": "path/to/Image_*.png", "count": 2}
            ]
        output_gif_path (str): Путь для сохранения GIF файла
        duration (int): Длительность каждого кадра в миллисекундах
        loop (int): Количество циклов (0 - бесконечный цикл)
    """
    
    # Собираем все кадры в правильном порядке
    all_frames = []
    
    for source in sources_sequence:
        pattern = source["pattern"]
        count = source.get("count", None)
        
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
    
    # Загружаем и обрабатываем изображения
    images = []
    print(f"Загрузка {len(all_frames)} изображений...")
    
    for i, image_path in enumerate(all_frames):
        try:
            # Открываем изображение с PIL
            img = Image.open(image_path)
            
            # Конвертируем в RGB если нужно (для совместимости с GIF)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Конвертируем обратно в палитровый режим для лучшего GIF качества
            img_rgb = img.convert('RGB')
            
            # Создаем палитровое изображение с уменьшенной палитрой
            # для уменьшения размера файла
            img_palette = img_rgb.convert('P', palette=Image.ADAPTIVE, colors=256)
            
            images.append(img_palette)
            
            if i % 10 == 0:
                print(f"Загружено {i+1}/{len(all_frames)}: {os.path.basename(image_path)}")
                
        except Exception as e:
            print(f"Ошибка при загрузке {image_path}: {e}")
            continue
    
    if not images:
        print("Не удалось загрузить ни одного изображения")
        return
    
    # Сохраняем как GIF
    print(f"Создание GIF из {len(images)} кадров...")
    
    # Сохраняем первое изображение и добавляем остальные
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    print(f"GIF сохранен как: {output_gif_path}")
    print(f"Размер: {os.path.getsize(output_gif_path) / 1024 / 1024:.2f} MB")

def create_gif_with_dynamic_sequence(source_patterns, sequence, output_gif_path, duration=100, loop=0):
    """
    Создает GIF с динамической последовательностью кадров из разных источников
    
    Args:
        source_patterns (dict): Словарь с шаблонами источников
            Пример: {
                "main": "path/to/Image_*.png",
                "passing": "path/to/Image_Passing_*.png"
            }
        sequence (list): Последовательность использования источников
            Пример: ["main", "main", "passing", "main", "passing"]
        output_gif_path (str): Путь для сохранения GIF файла
        duration (int): Длительность каждого кадра в миллисекундах
        loop (int): Количество циклов (0 - бесконечный цикл)
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
    frame_counters = {name: 0 for name in source_patterns.keys()}
    
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
    
    # Загружаем и обрабатываем изображения
    images = []
    for i, image_path in enumerate(all_frames):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_rgb = img.convert('RGB')
            img_palette = img_rgb.convert('P', palette=Image.ADAPTIVE, colors=256)
            images.append(img_palette)
        except Exception as e:
            print(f"Ошибка при загрузке {image_path}: {e}")
            continue
    
    # Сохраняем как GIF
    if images:
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=True
        )
        print(f"GIF сохранен как: {output_gif_path}")
        print(f"Размер: {os.path.getsize(output_gif_path) / 1024 / 1024:.2f} MB")
    else:
        print("Не удалось загрузить ни одного изображения для GIF")

# Примеры использования
if __name__ == "__main__":
    # Пример 1: Простая последовательность источников
    sources_sequence = [
        {"pattern": "ImageBlackHole/RotationAroundWormhole/Image_*.png", "count": 200},
        {"pattern": "ImageBlackHole/ImagePassing/Image_Passing_*.png", "count": 200},
        {"pattern": "ImageBlackHole/RotationCamera/Image_*.png", "count": 100},
        {"pattern": "ImageBlackHole/ImagePassingRotation/Image_*.png", "count": 200}
    ]
    
    create_gif_from_multiple_sources(
        sources_sequence, 
        "animation.gif", 
        duration=50,  # 50ms = 20 FPS
        loop=0        # бесконечный цикл
    )
    
    # # Пример 2: Динамическая последовательность
    # source_patterns = {
    #     "main": "RotationAroundWormhole/Image_*.png",
    #     "passing": "ImagePassing/Image_Passing_*.png"
    # }
    
    # # Чередующаяся последовательность
    # sequence = ["main"] * 3 + ["passing"] * 2 + ["main"] * 2 + ["passing"] * 1 + ["main"] * 4
    
    # create_gif_with_dynamic_sequence(
    #     source_patterns, 
    #     sequence, 
    #     "dynamic_animation.gif",
    #     duration=100,  # 100ms = 10 FPS
    #     loop=1         # 1 цикл
    # )