import numpy as np
import matplotlib.pyplot as plt

def plot_kerr_newman(M, Q, L):
    """
    Построение горизонтов и эргосфер черной дыры Керра-Ньюмена в полярных координатах
    
    Parameters:
    M - масса черной дыры
    Q - электрический заряд  
    L - момент импульса (угловой момент)
    """
    
    # Создаем полярную систему координат
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Углы от 0 до 2π
    theta = np.linspace(0, 2*np.pi, 500)
    
    print(f"Параметры черной дыры: M={M}, Q={Q}, L={L}")
    
    # Проверяем существование горизонтов
    horizon_param = M**2 - Q**2 - (L/M)**2
    print(f"Параметр горизонтов: {horizon_param}")
    
    # Отрисовываем эргосферы (только если есть вращение)
    if not L == 0:
        param_erg = M**2 - Q**2 - (L/M)**2 * np.cos(theta)**2
        valid_erg = param_erg >= 0
        
        if np.any(valid_erg):
            # Внешняя эргосфера
            r_erg_pl = np.where(valid_erg, M + np.sqrt(param_erg), np.nan)
            ax.plot(theta, r_erg_pl, 'g-', linewidth=3, alpha=0.8, label='Внешняя эргосфера (erg+)')
            
            # Внутренняя эргосфера  
            r_erg_in = np.where(valid_erg, M - np.sqrt(param_erg), np.nan)
            ax.plot(theta, r_erg_in, 'm-', linewidth=3, alpha=0.8, label='Внутренняя эргосфера (erg-)')
            print("Эргосферы отрисованы")
        else:
            print("Эргосферы не существуют при данных параметрах")
    else:
        print("Вращение отсутствует - эргосферы не отрисовываются")
    
    # Отрисовываем горизонты событий
    if horizon_param >= 0:
        # Внешний горизонт
        r_pl = M + np.sqrt(horizon_param)
        ax.plot(theta, np.full_like(theta, r_pl), 'r-', linewidth=3, alpha=0.8, label='Внешний горизонт (r+)')
        
        # Внутренний горизонт
        r_in = M - np.sqrt(horizon_param)
        ax.plot(theta, np.full_like(theta, r_in), 'purple', linewidth=3, alpha=0.8, label='Внутренний горизонт (r-)')
        print(f"Горизонты: внешний r={r_pl:.2f}, внутренний r={r_in:.2f}")
    else:
        print("Горизонты не существуют (голая сингулярность)")
    
    # Настройка полярного графика
    ax.set_theta_offset(np.pi/2)  # 0 градусов сверху
    ax.set_theta_direction(-1)    # по часовой стрелке
    
    # Настройка сетки
    ax.grid(True, alpha=0.4)
    
    # Легенда
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=14)
    
    # Заголовок
    ax.set_title(f'Характер горизонтов\nметрики Керра-Ньюмена\nM={M:.1f}, Q={Q:.1f}, L={L:.1f}', pad=20, fontsize=14)
    
    plt.tight_layout()
    plt.show()

# Примеры использования с разными параметрами
if __name__ == "__main__":
    '''
    print("=== Пример 1: Черная дыра Керра (вращение, без заряда) ===")
    plot_kerr_newman(M=1.0, Q=0.0, L=0.8)
    
    print("\n=== Пример 2: Черная дыра Райсснера-Нордстрёма (заряд, без вращения) ===")
    plot_kerr_newman(M=1.0, Q=0.6, L=0.0)
    
    print("\n=== Пример 3: Черная дыра Керра-Ньюмена (вращение + заряд) ===")
    plot_kerr_newman(M=1.0, Q=0.3, L=0.7)
    
    print("\n=== Пример 4: Шварцшильд (без заряда и вращения) ===")
    plot_kerr_newman(M=1.0, Q=0.0, L=0.0)
    
    print("\n=== Пример 5: Предельный случай (голая сингулярность) ===")
    plot_kerr_newman(M=1.0, Q=0.9, L=0.9)'''

    M: float = 2.2
    Q: float = 0.1
    L: float = (M-0.01)**2

    plot_kerr_newman(M=M, Q=Q, L=L)




    
