# EP ACH2016 - "IA" | TURMA 04
# BRUNO LEITE DE ANDRADE - 11369642
# FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
# GUILHERME DIAS JIMENES - 11911021
# IGOR AUGUSTO DOS SANTOS - 11796851
# LAURA PAIVA DE SIQUEIRA – 1120751

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
    