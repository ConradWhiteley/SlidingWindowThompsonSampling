from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np


class Environment(ABC):
    """
    Abstract class for a non-stationary environment
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Constructor for abstract Environment class
        """
        self.name = 'Abstract_Environment'
        self.rng = np.random.RandomState(random_seed)
        return

    @abstractmethod
    def reset(self):
        """
        Abstract class method to reset the environment
        :return:
        """
        pass


class StationaryEnvironment(Environment):
    """
    Class for a stationary environment
    """

    def __init__(self, mu_K: List[Union[float, int]], std_K: Union[float, int, List[Union[float, int]]] = 1,
                 random_seed: Optional[int] = None):
        """
        Constructor for a stationary environment
        :param mu_K: (List[Union[float, int]]) action mu values
        :param std_K: (Union[float, int, List[Union[float, int]]]) action std values
        :param random_seed: (int) random seed
        """
        super().__init__(random_seed)
        self.name = 'Stationary'
        self.K_actions = len(mu_K)
        self.mu_K = mu_K
        if isinstance(std_K, (float, int)):
            self.std_K = [std_K] * self.K_actions
        if len(mu_K) != len(self.std_K):
            raise ValueError('Length of mu_K and std_K are not equal.')
        return

    def simulate_environment(self, T: int, N_sims=10):
        """
        Simulate the environment to see what it would look like
        :param T: (int) total number of time steps
        :param N_sims: (int) number of simulations to perform
        :return: (np.ndarray) rewards of the simulations, of shape (K_action, N_sims, T)
        """
        rewards = np.full((self.K_actions, N_sims, T), np.nan)
        for k_action in range(self.K_actions):
            for i_sim in range(N_sims):
                rewards[k_action, i_sim] = self.rng.normal(loc=self.mu_K[k_action], scale=self.std_K[k_action], size=T)
        return rewards

    def sample_action(self, k_action: int):
        """
        Sample an action and update the environment
        :param k_action: (int) action selected
        :return:
        """
        reward_sample = self.rng.normal(loc=self.mu_K[k_action], scale=self.std_K[k_action])
        return reward_sample

    def ask_oracle(self):
        # optimal action is the one with the maximum expected reward
        opt_action = np.argmax(self.mu_K)
        opt_reward = self.mu_K[opt_action]  # equivalent to np.max(self.mu_K)
        return opt_action, opt_reward


class AbruptlyChangingEnvironment(Environment):
    """
    Class for an abruptly changing environment
    """

    def __init__(self):
        """
        Constructor for an abruptly changing environment
        """
        super().__init__()
        self.name = 'Abruptly_Changing'


class SmoothlyChangingEnvironment(Environment):
    """
    Class for a smoothly changing environment
    """

    def __init__(self):
        """
        Constructor for a smoothly changing environment
        """
        super().__init__()
        self.name = 'Smoothly_Changing'


class AbruptlyAndSmoothlyChangingEnvironment(Environment):
    """
    Class for a smoothly and abruptly changing environment
    """

    def __init__(self):
        """
        Constructor for an abruptly and smoothly changing environment
        """
        super().__init__()
        self.name = 'Abruptly_and_Smoothly_Changing'
        return
