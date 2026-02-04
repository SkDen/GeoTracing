import numpy as np
from PIL import Image

def create_chessboard_skybox(width=4096, height=2048, tile_size=45, 
                           color1=(255, 255, 255), color2=(0, 0, 0), 
                           line_color=(0, 0, 0), line_thickness=1):
    """
    Создает панорамное изображение SkyBox с шахматным паттерном
    
    Параметры:
    width, height: размер изображения (рекомендуется соотношение 2:1)
    tile_size: размер плитки в градусах
    color1, color2: цвета для шахматного паттерна
    line_color: цвет линий сетки
    line_thickness: толщина линий сетки в пикселях
    """
    # Создаем базовое изображение
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Вычисляем количество плиток по горизонтали и вертикали
    h_tiles = 360 // tile_size
    v_tiles = 180 // tile_size
    
    # Вычисляем размеры плитки в пикселях
    tile_width = width // h_tiles
    tile_height = height // v_tiles
    
    # Заполняем изображение шахматным паттерном
    for i in range(h_tiles):
        for j in range(v_tiles):
            # Определяем цвет плитки (шахматный порядок)
            color = color1 if (i + j) % 2 == 0 else color2
            
            # Вычисляем координаты плитки
            x_start = i * tile_width
            x_end = (i + 1) * tile_width
            y_start = j * tile_height
            y_end = (j + 1) * tile_height
            
            # Заполняем плитку цветом
            image[y_start:y_end, x_start:x_end] = color
    
    # Добавляем линии сетки
    if line_thickness > 0:
        # Вертикальные линии (меридианы)
        for i in range(h_tiles + 1):
            x = i * tile_width
            for t in range(line_thickness):
                current_x = (x + t) % width
                image[:, current_x] = line_color
        
        # Горизонтальные линии (параллели)
        for j in range(v_tiles + 1):
            y = j * tile_height
            for t in range(line_thickness):
                current_y = y + t
                if 0 <= current_y < height:
                    image[current_y, :] = line_color
    
    return Image.fromarray(image)

# Создаем и сохраняем изображение
chessboard_skybox = create_chessboard_skybox(
    width=4096,
    height=2048,
    tile_size=14,            # Размер плитки 30 градусов
    color1=(255, 255, 255),  # Белый цвет
    color2=(140, 140, 140),  # Светло-серый цвет
    line_color=(0, 0, 0),    # Черные линии
    line_thickness=10         # Толщина линий
)

chessboard_skybox.save('GeodesicEquations\\SkySpheres\\chessboard_skybox.png', 'PNG', dpi=(300, 300))

# Дополнительный вариант: создаем контрастный шахматный паттерн
contrast_skybox = create_chessboard_skybox(
    tile_size=15,           # Меньшие плитки
    color1=(0, 0, 0),       # Черный цвет
    color2=(255, 255, 255), # Белый цвет
    line_thickness=0        # Без линий
)

contrast_skybox.save('GeodesicEquations\\SkySpheres\\contrast_chessboard_skybox.png', 'PNG')
