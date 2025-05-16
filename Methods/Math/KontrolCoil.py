import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def ifft2c(k):
    return fft.ifftshift(fft.ifft2(fft.fftshift(k, axes=(0, 1)), norm='ortho'), axes=(0, 1))


def kontrol_coil(kspace):
    C = kspace.shape[-1]

    # IFFT по всем катушкам
    images = np.stack([ifft2c(kspace[:, :, c]) for c in range(C)])  # [C, H, W]

    # выбор опорной катушки (здесь: первая) - так как они одинаковые оказались по интенсивности
    ref_idx = 0
    I_ref = images[ref_idx]  # [H, W]

    sens_est = images / (I_ref + 1e-8)  # [C, H, W]

    # приведём к единичной мощности покатушечно
    power = np.sqrt(np.sum(np.abs(sens_est) ** 2, axis=0, keepdims=True))  # [1, H, W]
    sens_est /= (power + 1e-8)  # [C, H, W]

    return sens_est


def vizual(kspace, sens_ref):
    sens_est = kontrol_coil(kspace)
    cat_idx = 0
    true_map = sens_ref[:, :, cat_idx]
    est_map = sens_est[cat_idx]

    map1_1_KS = np.abs(true_map)
    map2_1_KS = np.abs(fft.fftshift(est_map))
    diff = np.abs(true_map - fft.fftshift(est_map))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(map1_1_KS, cmap='gray')
    plt.title("Истинная карта |S₀|")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(map2_1_KS, cmap='gray')
    plt.title("Оценка по опорной |Ŝ₀| (норм.)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='gray')
    plt.title("Разность")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Tk().withdraw()
    
    filename = askopenfilename(filetypes=[("NumPy files", "*.npy")], title="Выберите файл с картами чувствительности")
    sens_ref = np.load(filename)  # [высота, ширина, количество катушек]
    
    filename = askopenfilename(filetypes=[("NumPy files", "*.npy")], title="Выберите файл с к-пространством")
    kspace = np.load(filename)  # [высота, ширина, количество катушек]
    
    vizual(kspace, sens_ref)
