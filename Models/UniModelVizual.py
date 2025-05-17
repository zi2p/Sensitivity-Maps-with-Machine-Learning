from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.font import Font

from Models.BayesianRidge import BayesianRidgeModel
from Models.HistGradientBoosting import HistGBModel
from Models.KNeighborsRegressor import KNNModel
from Models.RandomForestRegressor import RandomForestModel
from Models.Ridge import RidgeModel


class SensitivityMapModel:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Sensitivity Maps")
        self.root.geometry("650x280")

        self.model_var = StringVar(value="Ridge")
        self.dataset_path = None
        self.test_kspace_path = None
        self.test_sens_path = None

        # выпадающий список
        temp_option = OptionMenu(root, self.model_var, "Ridge", "BayesianRidge", "RandomForest", "HistGB", "KNN")
        self.default_font = Font(font=temp_option.cget("font"))
        temp_option.destroy()

        self.italic_font = Font(font=self.default_font, slant="italic")
        self.normal_font = Font(font=self.default_font)

        self.btn_color = "#d3d3d3"  # светло-серый
        self.btn_fg = "black"
        self.btn_font = self.normal_font

        self.build_widgets()

    def build_widgets(self):
        (Label(self.root, text="Выберите модель:", font=self.normal_font)
         .grid(row=0, column=0, padx=10, pady=10, sticky="w"))
        model_menu = OptionMenu(self.root, self.model_var,
                                "Ridge",
                                "BayesianRidge",
                                "RandomForest",
                                "HistGB",
                                "KNN")
        model_menu.grid(row=0, column=1, columnspan=3, sticky="w")

        # датасет
        (Button(self.root, text="Выбрать файл датасета (.npz)", command=self.select_dataset,
                bg=self.btn_color, fg=self.btn_fg, font=self.btn_font)
         .grid(row=1, column=0, padx=10, pady=5, sticky="w"))
        self.lbl_dataset = Label(self.root, text="Файл не выбран", font=self.normal_font)
        self.lbl_dataset.grid(row=1, column=1, columnspan=3, sticky="w")

        # k-space 
        (Button(self.root, text="k-space для теста (.npy)", command=self.select_test_kspace,
                bg=self.btn_color, fg=self.btn_fg, font=self.btn_font)
         .grid(row=2, column=0, padx=10, pady=5, sticky="w"))
        self.lbl_test_kspace = Label(self.root, text="Файл не выбран", font=self.normal_font)
        self.lbl_test_kspace.grid(row=2, column=1, columnspan=3, sticky="w")

        # начать
        (Button(self.root, text="Обучение и тест", command=self.run,
                bg=self.btn_color, fg=self.btn_fg, font=self.btn_font)
         .grid(row=4, column=0, padx=10, pady=5, sticky="w"))

    def _update_label(self, label_widget, path):
        if path:
            label_widget.config(text=path, font=self.italic_font)
        else:
            label_widget.config(text="Файл не выбран", font=self.normal_font)

    def select_dataset(self):
        path = filedialog.askopenfilename(title="Выберите файл датасета (.npz)", filetypes=[("NPZ files", "*.npz")])
        self.dataset_path = path if path else None
        self._update_label(self.lbl_dataset, self.dataset_path)

    def select_test_kspace(self):
        path = filedialog.askopenfilename(title="Выберите k-space файл (.npy)", filetypes=[("NumPy files", "*.npy")])
        self.test_kspace_path = path if path else None
        self._update_label(self.lbl_test_kspace, self.test_kspace_path)

    def run(self):
        if not self.dataset_path:
            messagebox.showerror("Ошибка", "Выберите файл датасета (.npz)")
            return
        if not self.test_kspace_path:
            messagebox.showerror("Ошибка", "Выберите файл k-space для теста")
            return

        model_type = self.model_var.get()
        model_classes = {
            'Ridge': RidgeModel,
            'BayesianRidge': BayesianRidgeModel,
            'RandomForest': RandomForestModel,
            'HistGB': HistGBModel,
            'KNN': KNNModel
        }

        try:
            model_class = model_classes.get(model_type)
            if not model_class:
                raise ValueError("Выбранная модель не реализована")

            model = model_class(dataset_path=self.dataset_path)
            model.load_data()
            model.train()
            pred = model.predict_on_file(self.test_kspace_path, sens_path=self.test_sens_path)
            messagebox.showinfo("Готово", "Обучение и предсказание завершены!")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    root = Tk()
    app = SensitivityMapModel(root)
    root.mainloop()
