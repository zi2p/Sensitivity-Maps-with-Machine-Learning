import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

class KNNModel:
    def __init__(self, dataset_path='..\\Data\\dataset.npz', model_name='KNN', n_neighbors=5):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.n_neighbors = n_neighbors
        self.model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', n_jobs=-1))
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        print(f"Инициализирована модель {model_name} (n_neighbors={n_neighbors})")

    def load_data(self, test_size=0.2):
        data = np.load(self.dataset_path)
        X = data['X']
        Y = data['Y']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42)
        print(f"Данные загружены: {X.shape[0]} точек. \nТренировка: {self.X_train.shape[0]} \nТест: {self.X_test.shape[0]}")

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("KNN обучен.")

    def predict_on_file(self, kspace_path, sens_path=None, save_dir='..\\Data\\'):
        kspace = np.load(kspace_path)  # [H, W, C]
        H, W, C = kspace.shape
        pred_sens = np.zeros((H, W, C), dtype=np.complex64)

        for c in range(C):
            x = np.stack([
                kspace[:, :, c].real.flatten(),
                kspace[:, :, c].imag.flatten()
            ], axis=1)
            y_pred = self.model.predict(x)
            pred = y_pred[:, 0] + 1j * y_pred[:, 1]
            pred_sens[:, :, c] = pred.reshape(H, W)

        base = os.path.basename(kspace_path).replace(".npy", "")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{self.model_name.lower()}_map_{base}_{timestamp}.npy"
        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_path, pred_sens)
        print(f"Карта сохранена: {save_path}")

        if sens_path:
            true_map = np.load(sens_path)
            self.show_comparison(true_map, pred_sens)

    def show_comparison(self, true_map, pred_map):
        c = 0
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(true_map[:, :, c]), cmap='gray')
        plt.title("Истинная карта |S₀|")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(pred_map[:, :, c]), cmap='gray')
        plt.title(f"Предсказание {self.model_name}")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(true_map[:, :, c] - pred_map[:, :, c]), cmap='gray')
        plt.title("Разность")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        
# model = KNNModel(n_neighbors=5)
# model.load_data()
# model.train()
# model.predict_on_file('..\\Data\\k_space_1.npy', '..\\Data\\sens_maps_1.npy')
