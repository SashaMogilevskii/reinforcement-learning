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
        self.linear_1 = nn.Linear(state_dim, 64)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3 = nn.Linear(64, action_dim)
        self.activation = nn.ReLU()

    def forward(self, states):
        hidden = self.linear_1(states)
        hidden = self.activation(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.activation(hidden)
        actions = self.linear_3(hidden)
        return actions


class DQN_HardTargetUpdate:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01,
                 epilon_min=0.01, update_time=50):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.q_fucntion_second = Qfunction(self.state_dim, self.action_dim)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = deque(maxlen=15000)
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)
        self.counter = 0
        self.update_time = update_time

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

            # Использую вторую сеть
            targets = rewards + self.gamma * torch.max(self.q_fucntion_second(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()

            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease

            # Добавим время обновления весов в второй сети(Каждые 30 итераций)

            self.counter += 1  # Увеличение счетчика итераций
            if self.counter >= self.update_time:  # Проверка условия для обновления целевой сети
                self.q_fucntion_second.load_state_dict(self.q_function.state_dict())
                self.counter = 0




class DQN_SoftTargetUpdate:
    def __init__(self, state_dim, action_dim, coeff, gamma=0.99, lr=1e-2, batch_size=64, epsilon_decrease=0.01,
                 epilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.q_function_second = Qfunction(self.state_dim, self.action_dim)
        self.q_function_second.load_state_dict(
            self.q_function.state_dict()
        )
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = deque(maxlen=10000)
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)
        self.coefficient = coeff

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

            # update weight
            for target_param, local_param in zip(self.q_function_second.parameters(), self.q_function.parameters()):
                target_param.data.copy_(self.coefficient * local_param.data +
                                        (1.0 - self.coefficient) * target_param.data
                                        )


if __name__ == '__main__':

    # # Hard
    #
    # params = {"state_dim": state_dim,
    #           "action_dim": action_dim,
    #           "gamma": 0.99,
    #           "lr": 1e-3,
    #           "batch_size": 64,
    #           "epsilon_decrease": 0.03,
    #           "epilon_min": 0.03,
    #           "update_time": update_time}
    # agent = DQN_HardTargetUpdate(
    #     **params
    # )

    #Soft
    # params = {"state_dim": state_dim,
    #           "action_dim": action_dim,
    #           "gamma": 0.99,
    #           "coeff": 0.001,
    #           "lr": 1e-3,
    #           "batch_size": 64,
    #           "epsilon_decrease": 0.01,
    #           "epilon_min": 0.01}
    # agent = DQN_SoftTargetUpdate(
    #     **params
    # )


    for coef in [0.001, 0.005, 0.01, 0.02, 0.015, 0.1]:
        env = gym.make('LunarLander-v2')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        params = {"state_dim": state_dim,
                  "action_dim": action_dim,
                  "gamma": 0.99,
                  "coeff": coef,
                  "lr": 1e-3,
                  "batch_size": 64,
                  "epsilon_decrease": 0.01,
                  "epilon_min": 0.01}
        agent = DQN_SoftTargetUpdate(
            **params
        )

        episode_n = 100
        t_max = 500

        total_rewards_lst = []
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
            if episode % 10 == 0:
                print(f'episode: {episode}, total_reward: {total_reward}')
            total_rewards_lst.append(total_reward)

        print('total rewards lst', total_rewards_lst)

        print('max', max(total_rewards_lst))
        print(coef)
        print('----------')
