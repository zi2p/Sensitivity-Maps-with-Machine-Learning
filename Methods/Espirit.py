'''
    Алгоритм ESPIRiT был взят за готовый, реализацией алгоритмы занимался студент tg: @sg7reborn
    в качестве проектной работы по Физическим основам компьютерных и сетевых технологий.
'''

from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft


# Определение функций FFT и IFFT для многоканальных данных
def fft_shifted(x, axes):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm='ortho'), axes=axes)


def ifft_shifted(X, axes):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=axes), axes=axes, norm='ortho'), axes=axes)


# Адаптированный алгоритм ESPIRiT для 2D данных
def espirit_2d(X, k, r, t, c):
    """
    Derives the ESPIRiT operator for 2D k-space data.

    Arguments:
      X: Multi-channel 2D k-space data. Expected dimensions are (sx, sy, nc).
      k: Размер ядра k-пространства (например, 6).
      r: Размер калибровочной области (например, 24).
      t: Порог для отбора сингулярных значений (например, 0.01).
      c: Порог для отсечения собственных значений (например, 0.9925).

    Returns:
      maps: Оператор ESPIRiT с размерами (sx, sy, nc, nc).
    """

    sx, sy, nc = X.shape

    # Определение границ калибровочной области
    sxt = (sx // 2 - r // 2, sx // 2 + r // 2) if sx > 1 else (0, 1)
    syt = (sy // 2 - r // 2, sy // 2 + r // 2) if sy > 1 else (0, 1)

    # Извлечение калибровочной области
    C = X[sxt[0]:sxt[1], syt[0]:syt[1], :].astype(np.complex64)

    # Построение матрицы Ханкеля
    p = 2  # Для 2D данных
    num_blocks = (r - k + 1) ** p
    block_size = (k ** p) * nc
    A = np.zeros((num_blocks, block_size), dtype=np.complex64)

    idx = 0
    for xdx in range(r - k + 1):
        for ydx in range(r - k + 1):
            block = C[xdx:xdx + k, ydx:ydx + k, :].astype(np.complex64)
            A[idx, :] = block.flatten()
            idx += 1

    # Разложение по сингулярным значениям
    U, S, VH = np.linalg.svd(A, full_matrices=False)
    V = VH.conj().T

    # Отбор ядров на основе порога t
    n = np.sum(S >= t * S[0])
    V = V[:, :n]

    # Определение границ для ядра
    kxt = (sx // 2 - k // 2, sx // 2 + k // 2) if sx > 1 else (0, 1)
    kyt = (sy // 2 - k // 2, sy // 2 + k // 2) if sy > 1 else (0, 1)

    # Преобразование в k-пространственные ядра
    kernels = np.zeros((sx, sy, nc, n), dtype=np.complex64)
    kerdims = [k, k, nc]
    for idx in range(n):
        kernels[kxt[0]:kxt[1], kyt[0]:kyt[1], :, idx] = V[:, idx].reshape(kerdims)

    # Преобразование ядер в пространственное представление
    axes = (0, 1)
    kerimgs = np.zeros((sx, sy, nc, n), dtype=np.complex64)
    for idx in range(n):
        for jdx in range(nc):
            ker = np.flip(kernels[:, :, jdx, idx], axis=(0, 1)).conj()
            kerimgs[:, :, jdx, idx] = fft_shifted(ker, axes) * np.sqrt(sx * sy) / np.sqrt(k ** p)

    # Разложение на собственные значения и отсечение
    maps = np.zeros((sx, sy, nc, nc), dtype=np.complex64)
    for ix in range(sx):
        for iy in range(sy):
            Gq = kerimgs[ix, iy, :, :]
            u, s, vh = np.linalg.svd(Gq, full_matrices=False)
            for ldx in range(nc):
                if s[ldx] ** 2 > c:
                    maps[ix, iy, :, ldx] = u[:, ldx]

    return maps


# Определение прямого оператора
def forward_op(x, S, M):
    """
    Forward operator A(x) = M * F * (S * x)
    x: image (sx, sy)
    S: sensitivity maps (sx, sy, nc)
    M: sampling mask (sx, sy)
    Returns y: k-space data (sx, sy, nc)
    """
    # Умножение на карты чувствительности
    x_sens = x[:, :, np.newaxis] * S  # (sx, sy, nc)
    # Применение Фурье-преобразования
    X_sens = fft_shifted(x_sens, axes=(0, 1))  # (sx, sy, nc)
    # Применение маски выборки
    y = M[:, :, np.newaxis] * X_sens
    return y


# Определение сопряженного оператора
def adjoint_op(y, S, M):
    """
    Adjoint operator A^H(y) = sum over channels of conj(S) * F^H(M * y)
    y: k-space data (sx, sy, nc)
    S: sensitivity maps (sx, sy, nc)
    M: sampling mask (sx, sy)
    Returns x: image (sx, sy)
    """
    # Применение маски выборки
    y_masked = M[:, :, np.newaxis] * y
    # Обратное Фурье-преобразование
    x_sens = ifft_shifted(y_masked, axes=(0, 1))  # (sx, sy, nc)
    # Умножение на сопряженные карты чувствительности и суммирование по каналам
    x = np.sum(x_sens * np.conj(S), axis=2)
    return x


# Оператор полной вариации и его проксимальный оператор
def gradient(x):
    """
    Computes the gradient of x using forward differences.
    Returns gx and gy.
    """
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:-1, :] = x[1:, :] - x[:-1, :]
    gy[:, :-1] = x[:, 1:] - x[:, :-1]
    return gx, gy


def divergence(gx, gy):
    """
    Computes the divergence of the gradient field.
    """
    fx = np.zeros_like(gx)
    fy = np.zeros_like(gy)
    fx[1:, :] = gx[1:, :] - gx[:-1, :]
    fy[:, 1:] = gy[:, 1:] - gy[:, :-1]
    div = fx + fy
    return div


def tv_prox(x, tau):
    """
    Proximal operator for TV norm using Chambolle's algorithm.
    """
    px = np.zeros_like(x)
    py = np.zeros_like(x)
    tol = 1e-5
    maxiter = 100

    gx, gy = gradient(x)
    norm = np.sqrt(gx ** 2 + gy ** 2)
    d = np.maximum(1, norm / tau)
    px = gx / d
    py = gy / d

    for i in range(maxiter):
        div_p = divergence(px, py)
        diff = x - tau * div_p
        gx_new, gy_new = gradient(diff)
        norm_new = np.sqrt(gx_new ** 2 + gy_new ** 2)
        d = np.maximum(1, norm_new / tau)
        px_new = gx_new / d
        py_new = gy_new / d

        err = np.sum((px_new - px) ** 2 + (py_new - py) ** 2)
        px, py = px_new, py_new

        if err < tol:
            break

    div_p = divergence(px, py)
    return x - tau * div_p


def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


# Основной код
def main():
    # Загрузка данных k-пространства из 'k_space_undersampled.npy'
    X = np.load('k_space_undersampled.npy')  # Убедитесь, что файл находится в той же директории

    # Проверка размеров данных
    if X.ndim != 3:
        raise ValueError(f"Ожидаются 3 измерения (sx, sy, nc), но получено {X.ndim} измерений.")
    sx, sy, nc = X.shape
    print(f"Размеры k-пространства: sx={sx}, sy={sy}, nc={nc}")

    # Обратное FFT для получения данных в пространственном представлении
    x = ifft_shifted(X, axes=(0, 1))
    # np.sqrt(np.sum(np.abs(proj)**2, axis=2))
    x_compare = np.sqrt(np.sum(np.abs(ifft_shifted(np.load('k_space.npy'), axes=(0, 1))) ** 2, axis=2))

    # Параметры ESPIRiT (можно изменить при необходимости)
    k = 6  # Размер ядра k-пространства
    r = 16  # Размер калибровочной области
    t = 0.01  # Порог для сингулярных значений
    c = 0.9925  # Порог для собственных значений

    print("Вычисление оператора ESPIRiT...")
    esp = espirit_2d(X, k=k, r=r, t=t, c=c)

    # Отображение карт чувствительности
    print("Отображение карт чувствительности...")
    num_maps = min(esp.shape[3], 8)
    plt.figure(figsize=(12, 6))
    for i in range(num_maps):
        plt.subplot(2, 4, i + 1)
        plt.imshow(np.abs(esp[:, :, :, i]).sum(axis=2), cmap='gray')
        plt.title(f'Карта чувствительности {i + 1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Извлечение карт чувствительности из оператора ESPIRiT
    S = esp[:, :, :, 0]  # Берем первые карты чувствительности (sx, sy, nc)

    # Создание маски выборки из данных k-пространства
    M = (np.sum(np.abs(X), axis=2) > 0).astype(np.float32)

    # Определение правой части b = A^H y
    y = X  # Прореженные данные k-пространства
    b = adjoint_op(y, S, M)

    # Параметры для регуляризации
    lambda_tv = 0.01  # Параметр регуляризации по TV
    max_iter = 100  # Максимальное число итераций
    tol = 1e-6  # Точность

    # Инициализация
    x = np.zeros((sx, sy), dtype=np.complex64)
    t = 1
    x_old = x.copy()
    z = x.copy()

    # Предварительное вычисление липшицевой константы
    L = 1.0  # Можно оценить липшицеву константу или установить ее значение

    print("Выполнение реконструкции с регуляризацией по полной вариации...")
    for k_iter in range(max_iter):
        # Градиент ошибки
        grad = adjoint_op(forward_op(z, S, M) - y, S, M)

        # Обновление x
        x_new = z - (1 / L) * grad

        # Проксимальный шаг для TV
        x_new = tv_prox(x_new, lambda_tv / L)

        # Акселерация Нестерова
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)

        # Вычисление разницы для проверки сходимости
        diff = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
        if k_iter == 0:
            print(f"Итерация {k_iter}: относительная разница {diff:.5f}")
            # Отображение промежуточного результата
            plt.figure(figsize=(6, 6))
            plt.imshow(np.abs(x_new), cmap='gray')
            plt.title(f'Изображение до регуляризации')
            plt.axis('off')
            plt.show()

        if (k_iter % 10 == 0):
            print(f"Итерация {k_iter}: относительная разница {diff:.5f}")
            print(f"{calculate_psnr(x_new, x_compare)}")

        if diff < tol:
            print(f'Алгоритм сошелся на итерации {k_iter}')
            break

        x = x_new.copy()
        t = t_new

    # Финальное реконструированное изображение
    x_rec = x_new
    img = np.fft.ifftshift(x_rec, axes=(0, 1))
    # Отображение реконструированного изображения
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(img), cmap='gray')
    plt.title('Финальное реконструированное изображение')
    plt.axis('off')
    plt.show()


# ======= Использование =======

def espirit_maps(kspace):
    X = kspace  # [sx, sy, nc]

    maps = espirit_2d(X, k=10, r=64, t=0.01, c=0.9925)

    # Карты чувствительности из первого собственного значения
    sens_maps = maps[:, :, :, 0]  # [sx, sy, nc]
    return sens_maps


def vizual_diff(kspace, sens_maps):
    sens_maps_espirit = espirit_maps(kspace)
    
    map1 = np.abs((sens_maps[:, :, 0]))
    map2 = np.abs(fft.fftshift(sens_maps_espirit[:, :, 0]))
    diff = np.abs((sens_maps[:, :, 0]) - fft.fftshift(sens_maps_espirit[:, :, 0]))
    
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)

    plt.imshow(map1, cmap='gray')
    plt.title(f"Катушка {0}")
    plt.axis('off')

    plt.subplot(1, 3, 2)

    plt.imshow(map2, cmap='gray')
    plt.title(f"Катушка {0} ESPIRiT")
    plt.axis('off')

    plt.subplot(1, 3, 3)

    plt.imshow(diff, cmap='gray')
    plt.title(f"Разность")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def vizual(kspace):
    sens_maps = espirit_maps(kspace)

    # Визуализация одной карты
    plt.figure(figsize=(10, 4))
    for i in range(min(sens_maps.shape[-1], 4)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(np.abs(fft.fftshift(sens_maps[:, :, i])), cmap='gray')
        plt.title(f"Катушка {i}")
        plt.axis('off')
    plt.suptitle("ESPIRiT карты чувствительности")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Tk().withdraw()

    filename = askopenfilename(filetypes=[("NumPy files", "*.npy")], title="Выберите файл с картами чувствительности")
    sens_ref = np.load(filename)  # [высота, ширина, количество катушек]

    filename = askopenfilename(filetypes=[("NumPy files", "*.npy")], title="Выберите файл с к-пространством")
    kspace = np.load(filename)  # [высота, ширина, количество катушек]    

    vizual(kspace)
    vizual_diff(kspace, sens_ref)
