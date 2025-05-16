'''
    Файл предназначен для демонстрации реализации метода SOS.
    
    При запуске нужно будет выбрать 
                файл с картами чувствительности, 
                файл с к-пространством,
    затем будет продемонстирован результат для одной из катушек.
'''
from difflib import Differ

import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def ifft2c(k):
    return fft.ifftshift(fft.ifft2(fft.fftshift(k, axes=(0, 1)), norm='ortho'), axes=(0, 1))


def sum_of_squares(kspace):
    C = kspace.shape[-1]

    # IFFT по катушкам
    images = np.stack([ifft2c(kspace[:, :, c]) for c in range(C)])

    # модуль и норма
    denom = np.sqrt(np.sum(np.abs(images) ** 2, axis=0, keepdims=True)) + 1e-8
    sens_est = images / denom  # [количество катушек, высота, ширина], комплексные
    
    return sens_est


def vizual(kspace, sens_ref):
    sens_est = sum_of_squares(kspace)
    cat_idx = 0
    true_map = sens_ref[:, :, cat_idx]
    est_map = sens_est[cat_idx, :, :]
    
    map1_1_sos = np.abs(true_map)
    map2_1_sos = np.abs(fft.fftshift(est_map))
    diff = np.abs(true_map - fft.fftshift(est_map))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(map1_1_sos, cmap='gray')
    plt.title("Истинная карта |S₀|")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(map2_1_sos, cmap='gray')
    plt.title("Карта SOS |Ŝ₀|")
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
