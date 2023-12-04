import numpy as np
import random
import torch
import torch.nn as nn
import gym
from collections import deque

def set_seed(seed=1771):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed()


class Qfunction(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear_1 = nn.Linear(state_dim, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.linear_3 = nn.Linear(128, action_dim)
        self.activation = nn.ReLU()

    def forward(self, states):
        hidden = self.linear_1(states)
        hidden = self.activation(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.activation(hidden)
        actions = self.linear_3(hidden)
        return actions


class DQN:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-2, batch_size=64, epsilon_decrease=0.01,
                 epilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = deque(maxlen=15000)
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()

            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease



if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n


    # Hard
    params = {"state_dim": state_dim,
              "action_dim": action_dim,
              "gamma": 0.99,
              "lr": 0.002,
              "batch_size": 128,
              "epsilon_decrease": 0.025,
              "epilon_min": 0.025,
              }
    agent = DQN(
        **params
    )


    episode_n = 100
    t_max = 500
    lst_towards = []
    for episode in range(episode_n):
        total_reward = 0

        state = env.reset()
        for t in range(t_max):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            agent.fit(state, action, reward, done, next_state)

            state = next_state

            if done:
                break

        # print(f'episode: {episode}, memory_size: {len(agent.memory)}')
        print(f'episode: {episode}, total_reward: {total_reward}')
        lst_towards.append(total_reward)

    print(lst_towards)
    print(max(lst_towards))