from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Abstract class for an agent
    """

    def __init__(self):
        return

    @abstractmethod
    def select_action(self):
        """
        Abstract class method for the agent to select an action
        :return:
        """
        pass


class RandomAgent(Agent):
    """
    Class for a random agent
    """

    def __init__(self):
        """
        Constructor for random agent
        """
        super().__init__()
        self.name = 'Random_Agent'
        return

    def select_action(self):
        """
        Select an action with a random agent's policy
        :return:
        """
        raise NotImplementedError('Not yet implemented.')


class GreedyAgent(Agent):
    """
    Class for a greedy agent
    """

    def __init__(self, epsilon: int = 0):
        """
        Constructor for greedy agent
        :param epsilon: (int) the degree to which the agent is greedy
        """
        super().__init__()
        self.name = 'Greedy_Agent'
        return

    def select_action(self):
        """
        Select an action with a greedy agent's policy
        :return:
        """
        raise NotImplementedError('Not yet implemented.')


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
