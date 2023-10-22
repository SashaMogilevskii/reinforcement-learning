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

env = gym.make("MountainCarContinuous-v0")
action_dim = 1
state_dim = 2



class CrossEntropyAgent(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        self.action_n = action_dim
        self.state_dim = state_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        logit = self.forward(state)
        probs = self.Tanh(logit).detach().to('cpu').numpy()

        # action = np.random.choice(self.action_n, p=probs)

        return probs

    def fit(self, elit_trajectories):
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
            time.sleep(0.05)
            env.render()

        if done:
            break

    return trajectory

from tqdm import tqdm
def train_lopp():
    lst_rewards = []
    for iteration in range(iteration_n):

        # policy evaluation
        trajectories = [get_trajectory(env, agent, max_len=max_len ) for _ in tqdm(range(trajectory_n))]
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

agent = CrossEntropyAgent(state_dim=state_dim, action_dim=action_dim)
q_param = 0.6
trajectory_n = 20
iteration_n = 20
max_len = 999
lst_rew = train_lopp()
print(lst_rew)
traj = get_trajectory(env, agent, max_len=max_len, visualize=True)
print(traj['rewards'])




#
# print("initial state", env.reset())
# for i in range(10):
#     random_float = np.random.uniform(-1, 1)
#     random_float = np.array([random_float])
#     step = env.step(random_float)
#     print(f"action: {np.round(random_float, 3)} \t toward: {round(step[1], 4)}")
#
#     time.sleep(0.02)
#     env.render()
