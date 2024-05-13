from random import shuffle
import numpy as np
from typing import List
import numpy.typing as npt


def cross_validation_split(data: npt.ArrayLike, target: npt.ArrayLike, k: int):
    """
    Embaralha os dados em k folds para validação
    """

    # Embaralha os dados e os targets
    data_target = list(zip(data, target))
    shuffle(data_target)
    data, target = zip(*data_target)

    # Divide os dados em k folds
    data_folds = np.array_split(data, k)
    target_folds = np.array_split(target, k)

    return data_folds, target_folds
    