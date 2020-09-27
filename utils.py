"""Some useful functions."""
from typing import Union

import numpy as np


def _check_stochastic_input(n_arms: int) -> None:
    if not isinstance(n_arms, int):
        raise TypeError("n_arms must be an integer.")


def _check_update_input(chosen_arm: int, reward: Union[int, float]) -> None:
    if not isinstance(chosen_arm, (int, np.int64)):
        raise TypeError("chosen_arm must be an integer.")
    if not isinstance(reward, (int, float, np.int64, np.float64)):
        raise TypeError("reward must be an integer or a float.")
