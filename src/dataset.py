import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional


class FITS_Dataset(Dataset):
    """
    Класс для загрузки и обработки данных FITS-изображений.

    Args:
        data (np.ndarray): Массив NumPy, содержащий данные изображений.
                           Ожидается формат (N, H, W, C) или (N, C, H, W).
        transform (Optional[Callable]): Опциональная функция преобразования,
                                       применяемая к каждому образцу.
                                       По умолчанию None.
    """
    def __init__(self, data: np.ndarray, transform: Optional[Callable] = None):
        """
        Инициализирует набор данных.

        Args:
            data (np.ndarray): Массив данных.
            transform (Optional[Callable]): Преобразование данных.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Входные данные должны быть массивом NumPy.")

        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        """
        Возвращает общее количество образцов в наборе данных.

        Returns:
            int: Количество образцов.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Извлекает образец данных по указанному индексу.

        Args:
            idx (int): Индекс образца.

        Returns:
            torch.Tensor: Образец данных в виде тензора PyTorch.
        """
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return torch.from_numpy(sample).float()
