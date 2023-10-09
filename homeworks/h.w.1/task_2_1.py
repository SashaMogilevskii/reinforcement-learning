import time
import random

import gym
import numpy as np


def set_seed(seed=1771):
    random.seed(seed)
    np.random.seed(seed)


set_seed()

env = gym.make('Taxi-v3')

state_n = 500
action_n = 6


class CrossEntropyAgent():
    def __init__(self, state_n, action_n, lm):
        self.state_n = state_n
        self.action_n = action_n
        self.lm = lm
        """ Матрица состояний. в каждой строке будет вероятность каждого действия 
        то есть строка 0 - состояние 0, в котором есть 4 варианта действия 0, 1, 2, 3 - колонки. 
        Значение в матрице - оценка вероятности(вес?) данного действия в данном состояние """
        self.model = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(np.sum(new_model[state])) > 0:
                new_model[state] = (new_model[state] + self.lm) / (np.sum(new_model[state]) + self.lm * action_n)
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {"states": [],
                  "actions": [],
                  "rewards": []}

    state = env.reset()

    for _ in range(max_len):
        trajectory['states'].append(state)
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        obs, reward, done, _ = env.step(action)

        trajectory['rewards'].append(reward)

        state = obs

        if visualize:
            time.sleep(0.1)
            env.render()

        if done:
            break

    return trajectory


def train(q_param, trajectory_n, iteration_n, max_length, lm, state_n=500, action_n=6):
    agent = CrossEntropyAgent(state_n=state_n, action_n=action_n, lm=lm)
    mean_rewards_for_epochs = []
    for iteration in range(iteration_n):

        # policy evaluation
        trajectories = [get_trajectory(env, agent, max_len=max_length) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        print("iteration", iteration, 'mean total reward:', np.mean(total_rewards))
        mean_rewards_for_epochs.append((iteration, np.mean(total_rewards), max(total_rewards)))

        # policy impovement
        quantile = np.quantile(total_rewards, q_param)

        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        agent.fit(elite_trajectories)

    return {"q_param": q_param,
            "trajectory_n": trajectory_n,
            "iteration_n": iteration_n,
            "max_length": max_length,
            "info": mean_rewards_for_epochs}

for lm in [0, 0.01, 0.05, 0.1, 0.3, 0.5, 1]:
    results = train(q_param=0.8,
                    trajectory_n=400,
                    iteration_n=30,
                    max_length=300,
                    lm=lm
                    )
    print(lm)
    print(results)
    print("Средняя награда на последней эпохе:", results['info'][-1][1])
    print("Максимальная награда на последней эпохе:", results['info'][-1][2])
    print('-------------')
