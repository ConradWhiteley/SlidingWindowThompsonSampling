from typing import List, Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

from agents import RandomAgent, EpsilonGreedyAgent, Agent
from enviornments import StationaryEnvironment


def plot_simulated_environment(simulated_rewards: np.ndarray, step_size: int = 25):
    """
    Plot the sampled rewards from a simulated environment
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
    Plot the action history for each agent
    :param agents: (List[Agent]) agents
    :return:
    """
    fig, ax = plt.subplots(1, len(agents), figsize=(len(agents) * 7, 5))
    for i, agent in enumerate(agents):
        counts = agents[i].action_counts
        actions = np.arange(len(counts))
        ax[i].bar(actions, counts)
        ax[i].set_title(agent.name)
    fig.tight_layout()
    return fig


def plot_environment_setup(mu_df: pd.DataFrame, ax_lines: List[Optional[Union[float, int]]] = []):
    """
    Plot the true mean reward for each action
    :param mu_df: (pd.DataFrame) true action rewards
    :param ax_lines: (List[Union[Optional[float, int]]]) vertical lines to plot
    :return:
    """
    fig, ax = plt.subplots(figsize=(len(str(len(mu_df))) * 5, 8))
    melted_df = mu_df.reset_index().melt('t', var_name=['action'], value_name='mu')
    sns.scatterplot(data=melted_df, x='t', y='mu', hue='action', ax=ax)
    if len(ax_lines) > 0:
        for ax_line in ax_lines:
            ax.axvline(ax_line, linestyle='--', color='k')
    ax.set_xlabel('Time step, t')
    ax.set_title('True Mean Reward')
    ax.set_title('True Mean Reward for each Action Throughout the Experiment')
    return fig


def create_stationary_environment_setup(T: int, K_actions: int, mean: float = 3, std: float = 1,
                                        random_seed: Optional[int] = None):
    """
    Configure the true reward distribution for each action under an abruptly changing environment
    :param T: (int) total number of time steps
    :param K_actions: (int) number of actions
    :param mean: (float) mean of Normal distribution used to sample mean of actions' reward distributions
    :param std: (float) std of Normal distribution used to sample mean of actions' reward distributions
    :param random_seed: (int) random seed
    :return:
    """
    rng = np.random.RandomState(random_seed)
    mu_df = pd.DataFrame(index=np.arange(T), columns=[f'Action_{k_action}' for k_action in range(K_actions)])
    mu_df.index.name = 't'
    mu_K = rng.normal(loc=mean, scale=std, size=K_actions)
    mu_df.loc[:, :] = mu_K
    # plot environment setup
    plot_environment_setup(mu_df)
    return mu_K


def create_abruptly_changing_setup(T: int, K_actions: int, B_N: int, mean: float = 3, std: float = 1,
                                   random_seed: Optional[int] = None):
    """
    Configure the true reward distribution for each action under an abruptly changing environment
    :param T: (int) total number of time steps
    :param K_actions: (int) number of actions
    :param B_N: (int) number of breakpoints
    :param mean: (float) mean of Normal distribution used to sample mean of actions' reward distributions
    :param std: (float) std of Normal distribution used to sample mean of actions' reward distributions
    :param random_seed: (int) random seed
    :return:
    """
    if B_N > T:
        raise ValueError('Number of breakpoints is larger than total number of time steps.')
    mu_df = pd.DataFrame(index=np.arange(T), columns=[f'Action_{k_action}' for k_action in range(K_actions)])
    mu_df.index.name = 't'
    rng = np.random.RandomState(random_seed)
    B = np.sort(rng.choice(T, size=B_N, replace=False))
    prev_idx = 0
    for idx in list(B) + [T]:
        phase_mu_K = rng.normal(loc=mean, scale=std, size=K_actions)
        repeat_phase_mu_K = np.repeat(phase_mu_K.reshape(1, -1), idx - prev_idx, axis=0)
        mu_df.loc[prev_idx:idx - 1, :] = repeat_phase_mu_K
        prev_idx = idx
    # plot environment setup
    plot_environment_setup(mu_df, ax_lines=list(B - 0.5))
    return mu_df


def main():
    """
    Run the experiment
    :return:
    """
    T = 7500
    K_ACTIONS = 5
    B_N = 4
    RANDOM_SEED = 42

    # init environment
    #mu_df = create_stationary_environment_setup(T, K_ACTIONS, random_seed=RANDOM_SEED)
    mu_df = create_abruptly_changing_setup(T, K_ACTIONS, B_N, random_seed=RANDOM_SEED)

    # get mu for each action in array form
    mu_K = mu_df.to_numpy().T

    # init environment
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
