import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def ifft2c(k):
    return fft.ifftshift(fft.ifft2(fft.fftshift(k, axes=(0, 1)), norm='ortho'), axes=(0, 1))


def geometry_coil(kspace):
    C = kspace.shape[-1]

    # IFFT
    images = np.stack([ifft2c(kspace[:, :, c]) for c in range(C)])  # [C, H, W]

    # комбинированное изображение
    abs_images = np.abs(images)
    weights = abs_images / (np.sqrt(np.sum(abs_images ** 2, axis=0, keepdims=True)) + 1e-8)
    I_comb = np.sum(weights * images, axis=0)

    # только фаза
    phi = np.angle(I_comb)
    phase_factor = np.exp(-1j * phi)  # [H, W]

    # приводим фазу изображений
    images_phased = images * phase_factor  # [C, H, W]

    # нормализация по энергии (как в SOS)
    norm = np.sqrt(np.sum(np.abs(images) ** 2, axis=0, keepdims=True)) + 1e-8
    sens_est = images_phased / norm  # [C, H, W]

    return sens_est


def vizual(kspace, sens_ref):
    sens_est = geometry_coil(kspace)
    est_map = sens_est[0]
    true_map = sens_ref[:, :, 0]

    map1_1_geom = np.abs(true_map)
    map2_1_geom = np.abs(fft.fftshift(est_map))
    diff = np.abs(true_map - fft.fftshift(est_map))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(map1_1_geom, cmap='gray')
    plt.title("Истинная карта |S₀|")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(map2_1_geom, cmap='gray')
    plt.title("Взвешенное + норм. |Ŝ₀|")
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
