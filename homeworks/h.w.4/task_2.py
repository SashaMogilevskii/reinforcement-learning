import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2")


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def MonteCarlo(env, episode_n, trajectory_len=500, gamma=0.99):
    total_rewards = []

    qfunction = dict()
    counter = dict()
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n
        trajectory = {'states': [], 'actions': [], 'rewards': []}

        state = env.reset()
        state = tuple(np.round(state * 2) / 2)
        if state not in qfunction:
            qfunction[state] = np.zeros(action_n)
            counter[state] = np.zeros(action_n)

        for _ in range(trajectory_len):

            trajectory['states'].append(state)

            action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
            trajectory['actions'].append(action)

            state, reward, done, _ = env.step(action)
            state = tuple(np.round(state * 2) / 2)
            if state not in qfunction:
                qfunction[state] = np.zeros(action_n)
                counter[state] = np.zeros(action_n)

            trajectory['rewards'].append(reward)

            if done:
                break

        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1} is finished.')

        total_rewards.append(sum(trajectory['rewards']))

        real_trajectory_len = len(trajectory['rewards'])
        returns = np.zeros(real_trajectory_len + 1)
        for t in range(real_trajectory_len - 1, -1, -1):
            returns[t] = trajectory['rewards'][t] + gamma * returns[t + 1]

        for t in range(real_trajectory_len):
            state = trajectory['states'][t]
            action = trajectory['actions'][t]
            qfunction[state][action] += (returns[t] - qfunction[state][action]) / (1 + counter[state][action])
            counter[state][action] += 1

    return total_rewards


def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)
    qfunction = dict()

    for episode in range(episode_n):
        epsilon = 1 / (episode + 1)

        state = env.reset()
        state = tuple(np.round(state * 2) / 2)
        if state not in qfunction:
            qfunction[state] = np.zeros(action_n)

        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for _ in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(np.round(next_state * 2) / 2)
            if next_state not in qfunction:
                qfunction[next_state] = np.zeros(action_n)

            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)

            qfunction[state][action] += alpha * (
                    reward + gamma * qfunction[next_state][next_action] - qfunction[state][action])

            state = next_state
            action = next_action

            total_rewards[episode] += reward

            if done:
                break
        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1} is finished.')

    return total_rewards.tolist()


def QLearning(env, episode_n, trajectory_len, gamma=0.99, t_max=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)

    qfunction = dict()

    for episode in range(episode_n):
        epsilon = 1 / (episode + 1)

        state = env.reset()
        state = tuple(np.round(state * 2) / 2)
        if state not in qfunction:
            qfunction[state] = np.zeros(action_n)
        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for _ in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(np.round(next_state * 2) / 2)
            if next_state not in qfunction:
                qfunction[next_state] = np.zeros(action_n)

            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)

            qfunction[state][action] += alpha * (
                    reward + gamma * max(qfunction[next_state]) - qfunction[state][action])

            state = next_state
            action = next_action

            total_rewards[episode] += reward

            if done:
                break
        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1} is finished.')

    return total_rewards.tolist()


action_n = 4

total_rewards = SARSA(env, episode_n=1000, trajectory_len=1000, gamma=0.5)

print(total_rewards)
plt.plot(total_rewards)
plt.show()
