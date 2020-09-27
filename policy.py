from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from utils import (_check_stochastic_input,
                   _check_update_input)


class PolicyInterface(ABC):
    """Abstract Base class for all policies"""

    @abstractmethod
    def select_arm(self) -> int:
        """Select arms according to the policy for new data.
        Returns
        -------
        result: int
            The selected arm.
        """
        pass

    @abstractmethod
    def update(self, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update the reward information about each arm.
        Parameters
        ----------
        chosen_arm: int
            The chosen arm.
        reward: int, float
            The observed reward value from the chosen arm.
        """
        pass


class BasePolicy(PolicyInterface):
    """Base class for basic policies.
    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.
    """

    def __init__(self, n_arms: int) -> None:
        """Initialize class."""
        _check_stochastic_input(n_arms)

        self.n_arms = n_arms
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.cumulative_rewards = np.zeros(self.n_arms)
        self.t = 0

    def update(self, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update the reward information about each arm.
        Parameters
        ----------
        chosen_arm: int
            The chosen arm.
        reward: int, float
            The observed reward value from the chosen arm.
        """
        _check_update_input(chosen_arm, reward)

        self.t += 1
        self.counts[chosen_arm] += 1
        self.cumulative_rewards[chosen_arm] += reward


class UCB(BasePolicy):
    """Upper Confidence Bound.
    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.
    """

    def __init__(self, n_arms: int) -> None:
        """Initialize class."""
        super().__init__(n_arms)

    def select_arm(self) -> int:
        """Select arms according to the policy for new data.
        Returns
        -------
        arm: int
            The selected arm.
        """
        if 0 in self.counts:
            arm = np.argmin(self.counts)
        else:
            arm = np.argmax(self.mean_rewards + self.correction_factor)

        return arm

    @property
    def mean_rewards(self) -> np.ndarray:
        """
        Returns:
            numpy.ndarray: mean rewards each arm
        """
        return self.cumulative_rewards / self.counts

    @property
    def correction_factor(self) -> np.ndarray:
        """
        Returns:
            numpy.ndarray: correction factor each arm
        """
        return np.sqrt(np.log(self.t) / (2 * self.counts))


class UCBOffline(BasePolicy):
    def __init__(self, n_arms: int) -> None:
        """Initialize class."""
        _check_stochastic_input(n_arms, 1)
        super().__init__(n_arms)
        self.correction_factor_counts = np.zeros(self.n_arms, dtype=int)  # correction_factorのためのcounts

    def update(self, chosen_arm: int, rewards: np.ndarray) -> None:
        """
        Args:
            chosen_arm (int): selected arm
            rewards (numpy.ndarray): the observed rewards from selected arm. shape = (N, ), N is log size
        """
        if not isinstance(chosen_arm, (int, np.int64)):
            TypeError("chosen_arm must be int or numpy.int64")
        if not isinstance(rewards, np.ndarray):
            TypeError("rewards must be numpy.ndarray")
        if rewards.ndim != 1:
            TypeError("rewards must be 1 dim array")
        self.t += rewards.shape[0]
        self.counts[chosen_arm] += 1
        self.correction_factor_counts[chosen_arm] += rewards.shape[0]
        self.cumulative_rewards[chosen_arm] += rewards.mean()

    def select_arm(self) -> int:
        """Select arms according to the policy for new data.
        Returns
        -------
        arm: int
            The selected arm.
        """
        if 0 in self.counts:
            arm = np.argmin(self.counts)
        else:
            arm = np.argmax(self.mean_rewards + self.correction_factor)

        return arm

    @property
    def mean_rewards(self) -> np.ndarray:
        """
        Returns:
            numpy.ndarray: mean rewards each arm
        """
        return self.cumulative_rewards / self.counts

    @property
    def correction_factor(self) -> np.ndarray:
        """
        Returns:
            numpy.ndarray: correction factor each arm
        """
        return np.sqrt(np.log(self.t) / (2 * self.correction_factor_counts))


class UCB1(BasePolicy):
    """Upper Confidence Bound.
    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.
    """

    def __init__(self, n_arms: int) -> None:
        """Initialize class."""
        super().__init__(n_arms)

    def select_arm(self) -> int:
        """Select arms according to the policy for new data.
        Returns
        -------
        arm: int
            The selected arm.
        """
        if 0 in self.counts:
            arm = np.argmin(self.counts)
        else:
            arm = np.argmax(self.mean_rewards + self.correction_factor)

        return arm

    @property
    def mean_rewards(self) -> np.ndarray:
        """
        Returns:
            numpy.ndarray: mean rewards each arm
        """
        return self.cumulative_rewards / self.counts

    @property
    def correction_factor(self) -> np.ndarray:
        """
        Returns:
            numpy.ndarray: correction factor each arm
        """
        return np.sqrt(2 * np.log(self.t) / self.counts)
