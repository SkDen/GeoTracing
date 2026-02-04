import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from mpl_toolkits.mplot3d import Axes3D

def compute_surface(r0=1.0, m=1.0, a=0.5, l_min=-3, l_max=3, n_points=500, n_angles=100):
    """
    Построение поверхности вращения
    """
    
    # Создаем массив значений l
    l_arr = np.linspace(l_min, l_max, n_points)
    
    # Вычисляем r(l)
    cosh_ma = np.cosh(m * a)
    r_arr = r0 + (1/(2*m)) * np.log((np.cosh(m*(l_arr - a)) * np.cosh(m*(l_arr + a))) / (cosh_ma**2))
    
    # Вычисляем dz/dl
    tanh_sum = np.tanh(m*(l_arr - a)) + np.tanh(m*(l_arr + a))
    dz_dl = np.sqrt(1 - 0.25 * tanh_sum**2)
    
    # Численно интегрируем для получения z(l) - исправленная версия
    z_arr = cumulative_trapezoid(dz_dl, l_arr, initial=0)
    
    # Создаем угловую координату
    u_arr = np.linspace(0, 2*np.pi, n_angles)
    
    # Создаем сетку для параметров
    L, U = np.meshgrid(l_arr, u_arr)
    
    # Вычисляем R для сетки
    R = r0 + (1/(2*m)) * np.log((np.cosh(m*(L - a)) * np.cosh(m*(L + a))) / (cosh_ma**2))
    
    # Вычисляем Z для сетки (интерполируем z_arr на сетку)
    from scipy.interpolate import interp1d
    z_interp = interp1d(l_arr, z_arr, kind='cubic', fill_value='extrapolate')
    Z = z_interp(L)
    
    # Переходим к декартовым координатам
    X = R * np.cos(U)
    Y = R * np.sin(U)
    
    return X, Y, Z, l_arr, r_arr, z_arr


# Основное выполнение
if __name__ == "__main__":

    m = 0.05
    L = 0.0
    r0 = 2.0

    auto = False

    l = 200
    
    a = L/2
    
    # Вычисляем поверхность
    X, Y, Z, l_arr, r_arr, z_arr = compute_surface(r0=r0, a=a, m=m, l_min=-l, l_max=l)
    
    # Простая поверхность с выбором цветовой карты
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Можно экспериментировать с разными cmap:
    # 'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'Spectral', 'RdYlBu'
    surf = ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.7, 
                         linewidth=0.5, edgecolor='gray', antialiased=True)

    if auto:
        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-0.5, 5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Вложение параметризированной кротовой норы\nзначения параметров: L={L}, m={m}, r_0={r0}')
    # plt.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
    
    plt.show()
