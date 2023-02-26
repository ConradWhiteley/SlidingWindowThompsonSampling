from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np


class Agent(ABC):
    """
    Abstract class for an agent
    """

    def __init__(self, K_actions: int, random_seed: Optional[int] = None):
        """
        Constructor for abstract Agent class
        :param K_actions: (int) number of actions
        :param random_seed: (int) random seed
        """
        self.name = 'Abstract_Agent'
        self.K_actions = K_actions
        self.rng = np.random.RandomState(random_seed)
        self.reset()

    def reset(self) -> None:
        """
        Reset the agent's knowledge
        :return: (None)
        """
        # tracker for count and values
        self.action_counts = np.zeros(self.K_actions)
        self.reward_estimates = np.full(self.K_actions, np.nan)
        # memory to save sampled and optimal action and reward history
        init_size = 10000
        self.action_history = np.full(init_size, np.nan)
        self.reward_history = np.full(init_size, np.nan)
        self.opt_action_history = np.full(init_size, np.nan)
        self.opt_reward_history = np.full(init_size, np.nan)
        self.regret_history = np.full(init_size, np.nan)
        pass

    def random_tie_breaking_argmax(self, arr: np.ndarray) -> int:
        """
        Find index of maximum value in array, with random tie breaking
        - If all values are nan then randomly select from them
        - If some values are nan then randomly select from the maximum value and the nan values
        - If all are not nan then select the maximum value, with random tie-breaking
        :param arr: (np.ndarray) array
        :return: (int) argmax of array
        """
        if np.all(np.isnan(arr)):
            argmax = self.rng.randint(len(arr))
        elif np.any(np.isnan(arr)):
            nan_idx = np.argwhere(np.isnan(arr)).flatten()
            max_idx = np.argwhere(arr == np.nanmax(arr)).flatten()
            argmax = np.random.choice(np.append(nan_idx, max_idx))
        else:
            argmax = np.random.choice(np.flatnonzero(arr == np.max(arr)))
        return argmax

    def calculate_regret(self, reward: float, opt_reward: float) -> float:
        """
        Calculate regret: difference between optimal reward and sampled reward
        :param reward: (float) reward from action taken at time step t
        :param opt_reward: (float) optimal (maximum expected) reward at time step t
        :return: (float) regret of an action
        """
        regret = opt_reward - reward
        return regret

    def update_knowledge(self, t: int, action: int, reward: float, opt_action: int, opt_reward: float) -> None:
        """
        Update the agent's knowledge
        :param t: (int) time step
        :param action: (int) action taken at time step t
        :param reward: (float) reward from action taken at time step t
        :param opt_action: (int) optimal action to take at time step t
        :param opt_reward: (float) optimal (maximum expected) reward at time step t
        :return: (None)
        """
        # update count
        self.action_counts[action] += 1
        # update reward estimate
        prev_reward_estimate = self.reward_estimates[action]
        if np.isnan(prev_reward_estimate):
            self.reward_estimates[action] = reward
        else:
            error = reward - prev_reward_estimate
            self.reward_estimates[action] += error / self.action_counts[action]
        # update history
        self.action_history[t] = action
        self.reward_history[t] = reward
        self.opt_action_history[t] = opt_action
        self.opt_reward_history[t] = reward
        self.regret_history[t] = self.calculate_regret(reward, opt_reward)
        return

    @abstractmethod
    def select_action(self, t: int):
        """
        Abstract class method for the agent to select an action
        :param t: (int) time step
        :return:
        """
        pass


class RandomAgent(Agent):
    """
    Class for a random agent
    """

    def __init__(self, K_actions: int, random_seed: Optional[int] = None):
        """
        Constructor for a random agent
        :param K_actions: (int) number of actions
        :param random_seed: (int) random seed
        """
        super().__init__(K_actions, random_seed)
        self.name = 'Random_Agent'

    def select_action(self, t: int) -> None:
        """
        Select an action with a random policy
        :param t: (int) time step
        :return: (int) action
        """
        action = self.rng.choice(self.K_actions)
        return action


class EpsilonGreedyAgent(Agent):
    """
    Class for an epsilon-greedy agent
    """

    def __init__(self, K_actions: int, epsilon: Union[int, float] = 0, random_seed: Optional[int] = None):
        """
        Constructor for an epsilon greedy agent
        :param K_actions: (int) number of actions
        :param epsilon: (float) the probability the agent acts greedily
        :param random_seed: (int) random seed
        """
        super().__init__(K_actions, random_seed)
        if not (0 <= epsilon < 1):
            raise ValueError('Epsilon must be between [0,1)}.')
        self.epsilon = epsilon
        self.name = f'{self.epsilon}epsilon_Greedy_Agent' if epsilon > 0 else 'Greedy_Agent'
        return

    def select_action(self, t: int) -> int:
        """
        Select an action with an epsilon greedy policy
        :param t: (int) time step
        :return: (int) action
        """
        act_greedily = (self.rng.uniform() > self.epsilon)
        if act_greedily:
            next_action = self.random_tie_breaking_argmax(self.reward_estimates)
        else:
            next_action = self.rng.randint(len(self.reward_estimates))
        return next_action


class ThompsonSamplingAgent(Agent):
    """
    Class for a Thompson Sampling agent
    """

    def __init__(self):
        """
        Constructor for a Thompson Sampling agent
        """
        super().__init__()
        self.name = 'Thompson_Sampling_Agent'
        return

    def select_action(self):
        """
        Select an action with a Thompson Sampling agent's policy
        :return:
        """
        raise NotImplementedError('Not yet implemented.')


class SlidingWindowThompsonSamplingAgent(Agent):
    """
    Class for a Sliding-Window Thompson Sampling agent
    """

    def __init__(self):
        """
        Constructor for a Sliding-Window Thompson Sampling agent
        """
        super().__init__()
        self.name = 'Sliding_Window_Thompson_Sampling_Agent'
        return

    def select_action(self):
        """
        Select an action with a Sliding-Window Thompson Sampling agent's policy
        :return:
        """
        raise NotImplementedError('Not yet implemented.')


class UCBAgent(Agent):
    """
    Class for an UCB agent
    """

    def __init__(self):
        """
        Constructor for an UCB agent
        """
        super().__init__()
        self.name = 'UCB_Agent'
        return

    def select_action(self):
        """
        Select an action with an UCB agent's policy
        :return:
        """
        raise NotImplementedError('Not yet implemented.')


class LinearUCB(Agent):
    """
    Class for a Linear UCB agent
    """

    def __init__(self):
        """
        Select an action with a Linear UCB agent's policy
        """
        super().__init__()
        self.name = 'Linear_UCB_Agent'
        return

    def select_action(self):
        """
        Select an action with a Linear UCB agent's policy
        :return:
        """
        raise NotImplementedError('Not yet implemented.')
