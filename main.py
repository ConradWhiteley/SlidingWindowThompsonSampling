from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from agents import RandomAgent, EpsilonGreedyAgent, Agent
from enviornments import StationaryEnvironment


def plot_simulated_environment(simulated_rewards: np.ndarray, step_size: int = 25):
    """
    Plot the rewards from the simulated environment
    :param simulated_rewards: (np.ndarray) simulated reward with shape (K_actions, N_sims, T)
    :param step_size: (int) increments of the time steps to plot
    :return:
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    K_actions, N_sims, T = simulated_rewards.shape
    colours = plt.cm.rainbow(np.linspace(0, 1, K_actions))
    for k_action in range(K_actions):
        mean_reward = simulated_rewards[k_action].mean(axis=0)[::step_size]
        ax.plot(np.arange(0, T, step_size), mean_reward, color=colours[k_action], label=f'Action {k_action}')
    ax.legend()
    return fig


def plot_agent_action_history(agents: List[Agent]):
    """
    Plot the rewards from the simulated environment
    :param agents: (List[Agent]) agents
    :return:
    """
    fig, ax = plt.subplots(1, len(agents), figsize=(len(agents)*7, 5))
    for i, agent in enumerate(agents):
        counts = agents[i].action_counts
        actions = np.arange(len(counts))
        ax[i].bar(actions, counts)
        ax[i].set_title(agent.name)
    fig.tight_layout()
    return fig

def create_stationary_environment_setup(K_actions: int, mean: float=3, std: float = 1, random_seed: Optional[int] = None):
    """
    Configure the true reward distributions under an abruptly changing environment
    :param K_actions: (int) number of actions
    :param mean: (float) mean of Normal distribution used to sample mean of actions' reward distributions
    :param std: (float) std of Normal distribution used to sample mean of actions' reward distributions
    :param b: (int) number of breakpoints
    :return:
    """
    rng = np.random.RandomState(random_seed)
    mu_K = rng.normal(loc=mean, scale=std, size=K_actions)
    return mu_K

def create_abruptly_chaning_setup(K_actions: int, b: int):
    """
    Configure the true reward distributions under an abruptly changing environment
    :param K_actions: (int) number of actions
    :param b: (int) number of breakpoints
    :return:
    """
    raise NotImplementedError('Not implemented yet.')
    return

def main():
    """
    Run the experiment
    :return:
    """
    T = 5000
    K_ACTIONS = 4
    RANDOM_SEED = 42

    # init environment
    mu_K = create_stationary_environment_setup(K_ACTIONS, random_seed=RANDOM_SEED)
    env = StationaryEnvironment(mu_K, std_K=1, random_seed=RANDOM_SEED)

    # simulate the environment
    simulated_rewards = env.simulate_environment(T)
    plot_simulated_environment(simulated_rewards)

    # init agents
    agents = [
        RandomAgent(K_ACTIONS),
        EpsilonGreedyAgent(K_ACTIONS, epsilon=0),
        EpsilonGreedyAgent(K_ACTIONS, epsilon=0.01),
        EpsilonGreedyAgent(K_ACTIONS, epsilon=0.05),
        EpsilonGreedyAgent(K_ACTIONS, epsilon=0.5),
    ]

    # loop through time and agents
    # loop through time first so all agents operate on the same environment
    for t in tqdm(range(T)):
        for agent in agents:
            # select an action to take
            action = agent.select_action(t)

            # sample the environment and observe a reward
            reward = env.sample_action(action)

            # to calculate regret, ask oracle for optimal action and expected reward
            opt_action, opt_reward = env.ask_oracle()

            # update the agent's knowledge
            agent.update_knowledge(t, action, reward, opt_action, opt_reward)

    # plot action history of agents
    plot_agent_action_history(agents)

    return


if __name__ == '__main__':
    main()
