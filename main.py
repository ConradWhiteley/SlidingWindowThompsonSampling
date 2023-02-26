from typing import List

import matplotlib.pyplot as plt
import numpy as np
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
    fig, ax = plt.subplots(1, len(agents), figsize=(18, 5))
    for i, agent in enumerate(agents):
        actions, counts = np.unique(agents[i].action_history, return_counts=True)
        ax[i].bar(actions, counts)
        ax[i].set_title(agent.name)
    return fig


def main():
    """
    Run the experiment
    :return:
    """
    mu_K = [1, 1.25, 1.5, 1.75, 2]
    std_K = 0.1
    K_ACTIONS = len(mu_K)
    T = 5000
    RANDOM_SEED = 42

    # init environment
    env = StationaryEnvironment(mu_K, std_K=std_K, random_seed=RANDOM_SEED)

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
