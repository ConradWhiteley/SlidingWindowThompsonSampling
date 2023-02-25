from abc import ABC, abstractmethod


class NonStationaryEnvironment(ABC):
    """
    Abstract class for a non-stationary environment
    """

    def __init__(self):
        """
        Abstract class constructor
        """
        return

    @abstractmethod
    def reset(self):
        """
        Abstract class method to reset the environment
        :return:
        """
        pass


class AbruptlyChangingEnvironment(NonStationaryEnvironment):
    """
    Class for an abruptly changing environment
    """

    def __init__(self):
        """
        Constructor for an abruptly changing environment
        """
        super().__init__()
        return

    def reset(self):
        """
        Reset the abruptly changing environment
        :return:
        """
        raise NotImplementedError('Override.')


class SmoothlyChangingEnvironment(NonStationaryEnvironment):
    """
    Class for a smoothly changing environment
    """

    def __init__(self):
        """
        Constructor for a smoothly changing environment
        """
        super().__init__()
        self.name = 'UCB_Agent'
        return

    def reset(self):
        """
        Reset the abruptly changing environment
        :return:
        """
        raise NotImplementedError('Override.')


class AbruptlyAndSmoothlyChangingEnvironment(NonStationaryEnvironment):
    """
    Class for a smoothly and abruptly changing environment
    """

    def __init__(self):
        """
        Constructor for an abruptly and smoothly changing environment
        """
        super().__init__()
        return

    def reset(self):
        """
        Reset the abruptly and smoothly changing environment
        :return:
        """
        raise NotImplementedError('Override.')
