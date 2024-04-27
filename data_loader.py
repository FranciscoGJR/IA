from random import shuffle
import numpy as np
from typing import List
import numpy.typing as npt


def cross_validation_split(data: npt.ArrayLike, target: npt.ArrayLike, k: int):
    """
    Split data and target into k folds for cross validation.
    """

    # Shuffle data and target
    data_target = list(zip(data, target))
    shuffle(data_target)
    data, target = zip(*data_target)

    # Split data and target into k folds
    data_folds = np.array_split(data, k)
    target_folds = np.array_split(target, k)

    return data_folds, target_folds
    