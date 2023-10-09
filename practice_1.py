import gym
import gym_maze
import random
import time
import numpy as np

env = gym.make('maze-sample-5x5-v0')
state_n = 25

print("initial state", env.reset())
action = 0
print("step", env.step(action))


class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n

    def get_action(self, state):
        action = np.random.randint(self.action_n)
        return action


class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
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
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None


def get_state(obs):
    return int(np.sqrt(state_n) * obs[0] + obs[1])


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {"states": [],
                  "actions": [],
                  "rewards": []}

    obs = env.reset()
    state = get_state(obs)

    for _ in range(max_len):
        trajectory['states'].append(state)
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = get_state(obs)

        if visualize:
            time.sleep(0.1)
            env.render()

        if done: break

    return trajectory


agent = CrossEntropyAgent(state_n=25, action_n=4)
q_param = 0.9
trajectory_n = 50
iteration_n = 10

for iteration in range(iteration_n):

    # policy evaluation
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards'])for trajectory in trajectories]
    print("iteration", iteration, 'mean total reward:', np.mean(total_rewards))

    # policy impovement
    quantile = np.quantile(total_rewards, q_param)

    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)

# 36 min do conxa



trajectory = get_trajectory(env, agent, max_len=100, visualize=True)

print(agent.model)
