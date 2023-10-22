import time
import random

import torch
import gym
import numpy as np

from torch import nn
from tqdm import tqdm


def set_seed(seed=1771):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class CrossEntropyAgent(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        self.action_n = action_dim
        self.state_dim = state_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 0.3

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_n)
        ).to(self.device)

        self.Tanh = nn.Tanh()
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        self.loss = nn.MSELoss()
        self.t = 0

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state, iter):
        if iter < 10:
            action = np.array([TrainLoop.noise[0]])
            TrainLoop.noise = TrainLoop.noise[1:]


        else:
            self.network.eval()
            state = torch.FloatTensor(state).to(self.device)
            logit = self.forward(state)
            probs = self.Tanh(logit).detach().to('cpu').numpy()

            action = probs

        return action

    def fit(self, elit_trajectories):
        self.network.train()
        elite_states = []
        elit_actions = []
        for trajectory in elit_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                elite_states.append(state)
                elit_actions.append(action)

        elite_states = np.array(elite_states)
        elite_states = torch.FloatTensor(elite_states).to(self.device)

        pred_actions = self.forward(elite_states)
        elit_actions = torch.Tensor(np.array(elit_actions)).to(self.device)
        loss = self.loss(pred_actions, elit_actions)

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()


class TrainLoop():
    # fix seed, create env, create agent
    set_seed()

    def __init__(self,
                 iteration_n,
                 trajectory_n,
                 max_len,
                 q):

        self.env = gym.make("MountainCarContinuous-v0")
        self.env.seed(1771)
        self.agent = CrossEntropyAgent(action_dim=1,
                                       state_dim=2)
        self.iters = iteration_n  # count epochs
        self.trajectory_n = trajectory_n  # count samples in epochs
        self.q_param = q  # q_param
        self.max_steps = max_len  # max legnth each sample

    def get_trajectory(self, visualize=False, iter=0):

        if iter < 10:
            random_mean = np.random.uniform(-1,1)
            random_std_dev = np.random.uniform(0, 1)
            noise = np.random.normal(random_mean, random_std_dev, size=(self.max_steps,))
            noise = np.clip(noise, -1, 1)
            TrainLoop.noise = noise


        trajectory = {"states": [],
                      "actions": [],
                      "rewards": []}
        state = self.env.reset(seed=1771)
        for step in range(self.max_steps):

            trajectory['states'].append(state)
            action = self.agent.get_action(state, iter=iter)
            trajectory['actions'].append(action)

            state, reward, done, _ = self.env.step(action)

            trajectory['rewards'].append(reward)

            if visualize:
                time.sleep(0.05)
                self.env.render()

            if done:
                break

        self.agent.t = 0
        return trajectory

    def train(self):
        lst_rewards = []
        for iter in range(self.iters):

            if iter < 10:
                count_traj = 300
            else:
                count_traj =self.trajectory_n

            trajectories = []
            for traj in range(count_traj):
                self.result_traject = self.get_trajectory(iter=iter)
                trajectories.append(self.result_traject)

            total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]

            print("iteration", iter, 'mean total reward:', np.mean(total_rewards))
            lst_rewards.append(np.mean(total_rewards))
            # policy impovement
            elite_trajectories = []
            if iter > 10: # iterations were without noise
                quantile = np.quantile(total_rewards, self.q_param)
                for trajectory in trajectories:
                    total_reward = np.sum(trajectory['rewards'])
                    if total_reward > quantile:
                        elite_trajectories.append(trajectory)
            else:
                for trajectory in trajectories:
                    total_reward = np.sum(trajectory['rewards'])
                    if total_reward > 0:
                        print('nargrada', total_reward)
                        elite_trajectories.append(trajectory)

            if len(elite_trajectories) > 0:
                print(f'Кол-во элитных траекторий на которых обучается агент после шума на {iter} эпохи, {len(elite_trajectories)}')
                self.agent.fit(elite_trajectories)

        return lst_rewards


tr = TrainLoop(
    iteration_n=20,
    trajectory_n=20,
    max_len=1000,
    q=0.6)

lst_with_towards = tr.train()

print(tr.get_trajectory(visualize=True))
