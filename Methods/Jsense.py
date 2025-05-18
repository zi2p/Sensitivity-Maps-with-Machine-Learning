'''
Файл предназначен для демонстрации реализации метода JSENSE.

При запуске нужно будет выбрать:
    - файл с эталонными картами чувствительности,
    - файл с к-пространством ACS (опорное k-space),
    
затем будет выполнена оценка JSENSE и показаны карты чувствительности и реконструированное изображение.
'''

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def fft2c(x):
    return fft.fftshift(fft.fft2(fft.ifftshift(x, axes=(0, 1)), norm='ortho'), axes=(0, 1))


def ifft2c(x):
    return fft.fftshift(fft.ifft2(fft.ifftshift(x, axes=(0, 1)), norm='ortho'), axes=(0, 1))


# по методу SOS 
def init_maps(k_acs):
    C = k_acs.shape[2]
    imgs = np.stack([ifft2c(k_acs[:, :, c]) for c in range(C)], axis=-1)
    x0 = np.sqrt(np.sum(np.abs(imgs) ** 2, axis=-1))
    S0 = imgs / (x0[..., None] + 1e-8)
    return x0, S0


def update_image(k_acs, S):
    H, W, C = k_acs.shape
    num = np.zeros((H, W), dtype=np.complex64)
    den = np.zeros((H, W), dtype=np.float32)
    for c in range(C):
        y = k_acs[:, :, c]
        s = S[:, :, c]
        sx = ifft2c(y)
        num += np.conj(s) * sx
        den += np.abs(s) ** 2
    x = num / (den + 1e-8)
    return x


def update_maps(k_acs, x):
    H, W, C = k_acs.shape
    S_new = np.zeros((H, W, C), dtype=np.complex64)
    for c in range(C):
        y = k_acs[:, :, c]
        S_c = ifft2c(y) / (x + 1e-8)
        S_new[:, :, c] = S_c
    return S_new


def jsense(k_acs, num_iter=10):
    x, S = init_maps(k_acs)
    for i in range(num_iter):
        x = update_image(k_acs, S)
        S = update_maps(k_acs, x)
    return S, x


def vizual(S, x, sens_ref):
    C = S.shape[2]
    plt.figure(figsize=(15, 5))

    for c in range(C):  # полученные алгоритмом
        plt.subplot(4, C, c + 1)
        plt.imshow(np.abs(fft.fftshift(S[:, :, c])), cmap='gray')
        plt.title(f'|Ŝ_{c}|')
        plt.axis('off')

    for c in range(C):  # истинные
        plt.subplot(4, C, c + 1 + C)
        plt.imshow(np.abs((sens_ref[:, :, c])), cmap='gray')
        plt.title(f'|S_{c}|')
        plt.axis('off')

    for c in range(C):  # разность
        plt.subplot(4, C, c + 1 + C + C)
        plt.imshow(np.abs((sens_ref[:, :, c]) - fft.fftshift(S[:, :, c])), cmap='gray')
        plt.title(f'|S_{c}-Ŝ_{c}|')
        plt.axis('off')

    plt.subplot(4, 1, 4)  # изображение
    plt.imshow(np.abs(fft.fftshift(x)), cmap='gray')
    plt.title('Отреконструированное иображение')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Tk().withdraw()

    sens_file = askopenfilename(title="Выберите файл ИСТИННЫХ карт чувствительности (.npy)",
                                filetypes=[("NumPy files", "*.npy")])
    sens_ref = np.load(sens_file)  # [H, W, C]

    kspace_file = askopenfilename(title="Выберите ACS k-space файл (.npy)", filetypes=[("NumPy files", "*.npy")])
    k_acs = np.load(kspace_file)  # [H, W, C]

    S_est, x_est = jsense(k_acs, num_iter=10)
    vizual(S_est, x_est, sens_ref)
