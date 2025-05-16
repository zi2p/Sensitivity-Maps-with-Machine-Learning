import numpy.fft as fft
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import os

def mse(a, b):
    return np.mean((a - b) ** 2)

def compare_maps(gt_map, gt_name, method_maps):
    gt_abs = np.abs(gt_map[:, :, 0])  # Первая катушка

    plt.figure(figsize=(14, 3 + 3 * len(method_maps)))

    plt.subplot(len(method_maps)+1, 3, 2)
    plt.imshow(gt_abs, cmap='gray')
    plt.title(f"Истинная карта |{gt_name}|")
    plt.axis('off')

    for i, (name, map_arr) in enumerate(method_maps.items()):
        row = i + 2
        m_abs = np.abs(fft.fftshift(map_arr[:, :, 0]))
        diff = gt_abs - m_abs
        err = mse(gt_abs, m_abs)

        plt.subplot(len(method_maps)+1, 3, 3*(row-1) + 1)
        plt.imshow(m_abs, cmap='gray')
        plt.title(f"{name}")
        plt.axis('off')

        plt.subplot(len(method_maps)+1, 3, 3*(row-1) + 2)
        plt.imshow(diff, cmap='gray')
        plt.title(f"Разность (MSE={err:.5f})")
        plt.axis('off')

        plt.subplot(len(method_maps)+1, 3, 3*(row-1) + 3)
        plt.hist(diff.ravel(), bins=50, color='gray')
        plt.title("Гистограмма ошибки")

    plt.tight_layout()
    plt.show()


class CompareApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Сравнение карт чувствительности")

        self.gt_path = None
        self.gt_data = None
        self.gt_name = None

        self.maps_paths = []
        self.maps_data = {}

        tk.Button(master, text="Выбрать ИСТИННУЮ карту", width=30, command=self.load_gt).pack(pady=10)
        tk.Button(master, text="Выбрать до 4 карт для сравнения", width=30, command=self.load_maps).pack(pady=10)
        tk.Button(master, text="Сравнить", width=30, command=self.compare).pack(pady=10)

        self.status = tk.Label(master, text="Начать", fg="blue")
        self.status.pack(pady=5)

    def load_gt(self):
        path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")], title="Выберите ИСТИННУЮ карту")
        if path:
            self.gt_data = np.load(path)
            self.gt_path = path
            self.gt_name = os.path.basename(path)
            self.status.config(text=f"Истинная карта загружена: {self.gt_name}", fg="green")

    def load_maps(self):
        paths = filedialog.askopenfilenames(filetypes=[("NumPy files", "*.npy")], title="Выберите до 4 карт")
        if not paths:
            return
        if len(paths) > 4:
            messagebox.showwarning("Ограничение", "Можно выбрать не более 4 карт.")
            return
        self.maps_data = {}
        for path in paths:
            name = os.path.basename(path)
            self.maps_data[name] = np.load(path)
        self.status.config(text=f"Загружено {len(self.maps_data)} карт для сравнения", fg="green")

    def compare(self):
        if self.gt_data is None:
            messagebox.showerror("Ошибка", "Сначала выберите истинную карту.")
            return
        if not self.maps_data:
            messagebox.showerror("Ошибка", "Выберите хотя бы одну карту для сравнения.")
            return
        compare_maps(self.gt_data, self.gt_name, self.maps_data)


if __name__ == "__main__":
    root = tk.Tk()
    app = CompareApp(root)
    root.mainloop()
