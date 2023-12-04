import time
import random
import torch
from torch import nn
import gym
import numpy as np


def set_seed(seed=1771):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed()

env = gym.make("LunarLander-v2")
action_n = 4
state_n = 8


class CrossEntropyAgent(nn.Module):
    def __init__(self, action_n, state_dim):
        super().__init__()
        self.action_n = action_n
        self.state_dim = state_dim
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128,  64),
            nn.ReLU(),
            nn.Linear(64, self.action_n)
        )

        self.softmax = nn.Softmax()
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        probs = self.softmax(logits).data.numpy()

        action = np.random.choice(self.action_n, p=probs)

        return action

    def fit(self, elit_trajectories):
        elite_states = []
        elit_actions = []
        for trajectory in elit_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                elite_states.append(state)
                elit_actions.append(action)

        elite_states = torch.FloatTensor(elite_states)
        elit_actions = torch.LongTensor(elit_actions)

        pred_actions = self.forward(elite_states)

        loss = self.loss(pred_actions, elit_actions)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()


def get_trajectory(env, agent, max_len, visualize=False):
    trajectory = {"states": [],
                  "actions": [],
                  "rewards": []}

    state = env.reset()

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)

        trajectory['actions'].append(action)
        state, reward, done, _ = env.step(action)

        trajectory['rewards'].append(reward)

        if visualize:
            time.sleep(0.01)
            env.render()

        if done:
            break

    return trajectory


def train_lopp():
    lst_rewards = []
    for iteration in range(iteration_n):

        # policy evaluation
        trajectories = [get_trajectory(env, agent, max_len=max_len ) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        print("iteration", iteration, 'mean total reward:', np.mean(total_rewards))
        lst_rewards.append(np.mean(total_rewards))
        # policy impovement
        quantile = np.quantile(total_rewards, q_param)

        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        if len(elite_trajectories) > 0:
            agent.fit(elite_trajectories)

    return lst_rewards

agent = CrossEntropyAgent(state_dim=state_n, action_n=action_n)
q_param = 0.6
trajectory_n = 20
iteration_n = 20
max_len = 300
lst_rew = train_lopp()
print(lst_rew)
traj = get_trajectory(env, agent, max_len=max_len, visualize=True)
print(traj['rewards'])
