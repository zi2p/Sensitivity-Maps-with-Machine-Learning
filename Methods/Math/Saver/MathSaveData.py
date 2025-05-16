import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os

from Methods.Math.Sos import sum_of_squares
from Methods.Math.Phase import phase_consistency
from Methods.Math.Geometry import geometry_coil
from Methods.Math.KontrolCoil import kontrol_coil


def save_sensitivity_map(map_array, original_filename: str, method: str = "sos"):
    """
    Сохраняет карту чувствительности в папку ./Data/
    с именем: <method>_map_<basename>.npy
    """
    os.makedirs("../../Data", exist_ok=True)
    base = os.path.basename(original_filename)
    name = os.path.splitext(base)[0]  
    filename = f"{method}_map_{name}.npy"
    save_path = os.path.join("../../Data", filename)
    np.save(save_path, map_array)
    print(f"Карта чувствительности сохранена: {save_path}")


def run_method(method_name, kspace, kspace_path):
    if method_name == "sos":
        sens = sum_of_squares(kspace)
    elif method_name == "phase":
        sens = phase_consistency(kspace)
    elif method_name == "geometry":
        sens = geometry_coil(kspace)
    elif method_name == "kontrol":
        sens = kontrol_coil(kspace)
    else:
        raise ValueError("Неизвестный метод")


    sens = np.transpose(sens, (1, 2, 0))

    save_sensitivity_map(sens, kspace_path, method=method_name)
    messagebox.showinfo("Готово", f"Карта методом {method_name.upper()} сохранена!")


def select_kspace():
    return filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")], title="Выберите файл к-пространства")


def launch_interface():
    def select_and_run(method):
        kspace_path = select_kspace()
        if not kspace_path:
            return
        kspace = np.load(kspace_path)
        try:
            run_method(method, kspace, kspace_path)
        except Exception as e:
            messagebox.showerror("Ошибка выполнения", str(e))

    root = tk.Tk()
    root.title("Выбор метода генерации карты чувствительности")

    label = tk.Label(root, text="Выберите метод:")
    label.pack(pady=8)

    methods = {
        "Sum of Squares (SOS)": "sos",
        "Phase Consistency": "phase",
        "Geometry Coil": "geometry",
        "Kontrol Coil": "kontrol"
    }

    for label, key in methods.items():
        tk.Button(root, text=label, width=30, command=lambda m=key: select_and_run(m)).pack(pady=4)

    root.mainloop()


if __name__ == "__main__":
    launch_interface()
