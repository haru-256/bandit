"""This module contains bandit classes."""
from abc import abstractmethod
from typing import Optional, Union

import numpy as np


class BaseBandit:
    """Base class for all bandits.
    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.
    n_features: int
        The dimension of context vectors.
    noise: int, optional(default=None)
        The variance of gaussian noise on linear model of contextual rewards.
    contextual: bool, optional(default=False)
        Whether rewards are models contextual or not.
    """

    def __init__(self, n_arms: int, n_features: Optional[int] = None, scale: float = 0.1,
                 noise: float = 0.1, contextual: bool = False, mu: Optional[np.ndarray] = None) -> None:
        """Initialize Class."""
        self.rewards = 0.0
        self.regrets = 0.0
        self.n_arms = n_arms
        self.contextual = contextual
        if self.contextual:
            self.scale = scale
            self.noise = noise
            self.n_features = n_features
            self.params = np.random.multivariate_normal(np.zeros(self.n_features),
                                                        self.scale * np.identity(self.n_features), size=self.n_arms).T
        else:
            if mu is None:
                self.mu = np.random.uniform(low=0.001, high=0.1, size=n_arms)
            else:
                if not isinstance(mu, np.ndarray) or mu.shape != (self.n_arms,):
                    raise TypeError("mu must be numpy.ndarray.")
                self.mu = mu
            self.mu_max, self.best_arm = np.max(self.mu), np.argmax(self.mu)

    @abstractmethod
    def pull(self, chosen_arm: int, x: Optional[np.ndarray] = None) -> Union[int, float]:
        """Pull arms.
        chosen_arm: int
            The chosen arm.
        x : array-like, shape = (n_features, ), optional(default=None)
            A test sample.
        """
        pass


class BernoulliBandit(BaseBandit):
    """Bernoulli Bandit.
    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.
    n_features: int
        The dimension of context vectors.
    noise: float, optional(default=0.1)
        The variance of gaussian noise on linear model of contextual rewards.
    contextual: bool, optional(default=False)
        Whether rewards are models contextual or not.
    """

    def __init__(self, n_arms: int, n_features: Optional[int] = None, scale: float = 0.1,
                 noise: float = 0.1, contextual: bool = False, mu: Optional[np.ndarray] = None) -> None:
        """Initialize Class."""
        if contextual:
            raise NotImplementedError("contextual bandit has not implemented yet")
        super().__init__(n_arms=n_arms, n_features=n_features, scale=scale, noise=noise, contextual=contextual, mu=mu)

    def pull(self, chosen_arm: int, x: Optional[np.ndarray] = None) -> Union[int, float]:
        """Pull arms.
        chosen_arm: int
            The chosen arm.
        x : array-like, shape = (n_features, ), optional(default=None)
            A test sample.
        """
        if chosen_arm not in range(self.n_arms):
            raise ValueError(f"chosen_arm is not in range({self.n_arms})")
        reward, regret = \
            np.random.binomial(n=1, p=self.mu[chosen_arm]), self.mu_max - self.mu[chosen_arm]

        self.rewards += reward
        self.regrets += regret

        return reward
