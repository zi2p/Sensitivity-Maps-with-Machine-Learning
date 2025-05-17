import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_dataset(save_path='..\\Data\\dataset.npz', k_prefix='k_space_', s_prefix='sens_maps_', n=4):
    '''
    Собирает данные из всех файлов k_space_i.npy и sens_maps_i.npy
    и сохраняет их в единый .npz-файл для будущей загрузки.

    Parameters:
        save_path: путь сохранения .npz
        k_prefix: префикс файлов с к-пространством
        s_prefix: префикс файлов с картами
        n: количество пар файлов (i от 1 до n)
    '''
    
    X_all = []
    Y_all = []

    for i in range(1, n + 1):
        k = np.load(f'..\\Data\\{k_prefix}{i}.npy')        # [H, W, C]
        s = np.load(f'..\\Data\\{s_prefix}{i}.npy')        # [H, W, C]
        H, W, C = k.shape
        for c in range(C):
            x = np.stack([k[:, :, c].real.flatten(), k[:, :, c].imag.flatten()], axis=1)  # [H*W, 2]
            y = np.stack([s[:, :, c].real.flatten(), s[:, :, c].imag.flatten()], axis=1)  # [H*W, 2]
            X_all.append(x)
            Y_all.append(y)

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    np.savez_compressed(save_path, X=X_all, Y=Y_all)
    print(f"Данные сохранены в {save_path}: {X_all.shape[0]} векторов.")

def load_and_split_dataset(npz_path='..\\Data\\dataset.npz', test_size=0.2, random_state=42):
    """
    Загружает данные из .npz архива и делит их на обучающую и тестовую выборку.

    Returns:
        X_train, X_test, y_train, y_test
    """
    data = np.load(npz_path)
    X = data['X']
    Y = data['Y']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    print(f"Загружено: {X.shape[0]} векторов. \nТренировка: {len(X_train)} \nТест: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def inspect_dataset(path='..\\Data\\dataset.npz'):
    data = np.load(path)

    print("Ключи в архиве:", list(data.keys()))

    X = data['X']
    Y = data['Y']

    print(f"X (вход): форма = {X.shape}; структура: [Re(k), Im(k)]")
    print(f"Y (выход): форма = {Y.shape}; структура: [Re(sens), Im(sens)]")
        
# prepare_dataset(save_path="..\\Data\\dataset_cropp.npz", k_prefix='k_space_cropped_', s_prefix='sens_maps_', n=4)
# X_train, X_test, y_train, y_test = load_and_split_dataset()
# inspect_dataset()