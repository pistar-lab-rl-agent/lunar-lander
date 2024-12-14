"""
PPO algorithm.

ref
- https://github.com/bentrevett/pytorch-rl/blob/master/5a%20-%20Proximal%20Policy%20Optimization%20(PPO)%20%5BLunarLander%5D.ipynb
"""
import random
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import wandb
from original_dqn import AbsAgent


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, is_prelu=False):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU() if is_prelu else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU() if is_prelu else nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class PPOModel(nn.Module):
    def __init__(self, state_size: int, action_size: int, dropout=0.1, is_prelu=False):
        super(PPOModel, self).__init__()
        self.actor = MLP(state_size, 128, action_size, dropout, is_prelu)
        self.critic = MLP(state_size, 128, 1, dropout, is_prelu)

    def forward(self, x: torch.Tensor):
        action_pred = self.actor(x)
        value_pred = self.critic(x)
        return action_pred, value_pred


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values

    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


class PPOAgent(AbsAgent):
    def __init__(
            self, state_size: int, action_size: int, discount_factor: float = 0.98, learning_rate: float = 0.0005,
            eps_clip: float = 0.2, k_epoch: int = 3, dropout_rate: float = 0.0, is_prelu: bool = False,
            is_custom_init: bool = False,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.values = []
        self.rewards = []

        self.step_counter = 0

        # Hyperparameters
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.eps_clip = eps_clip
        self.k_epoch = k_epoch

        self.ppo_model = PPOModel(self.state_size, self.action_size, dropout_rate, is_prelu)
        if is_custom_init:
            self.ppo_model.apply(init_weights)
        self.optimizer = optim.Adam(self.ppo_model.parameters(), lr=self.learning_rate)

    def append_transition(self, state, action, log_prob_action, value, reward):
        self.states.append(torch.FloatTensor(state).unsqueeze(0))
        self.actions.append(action)
        self.log_prob_actions.append(log_prob_action)
        self.values.append(value)
        self.rewards.append(reward)

    def get_action(self, state: np.ndarray):
        self.step_counter += 1
        with torch.no_grad():
            action_pred, value_pred = self.ppo_model(torch.FloatTensor(state).unsqueeze(0))
            action_prob = F.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)
            return action, log_prob_action, value_pred

    def train(self, states, actions, log_prob_actions, advantages, returns):
        states = states.detach()
        actions = actions.detach()
        log_prob_actions = log_prob_actions.detach()
        advantages = advantages.detach()
        returns = returns.detach()

        for _ in range(self.k_epoch):
            # get new log prob of actions for all input states
            action_pred, value_pred = self.ppo_model(states)
            value_pred = value_pred.squeeze(-1)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)

            # new log prob using old actions
            new_log_prob_actions = dist.log_prob(actions)

            policy_ratio = (new_log_prob_actions - log_prob_actions).exp()

            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - self.eps_clip, max=1.0 + self.eps_clip) * advantages

            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.smooth_l1_loss(returns, value_pred).mean()

            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, num_episodes=5, render=False) -> Tuple[float, float]:
        env = gym.make('LunarLander-v3', render_mode='human' if render else None)
        self.ppo_model.eval()
        evaluation_scores = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            score = 0
            done = False
            truncated = False
            while not (done or truncated):
                action, *_ = self.get_action(state)
                next_state, reward, done, truncated, _ = env.step(action.item())
                score += reward
                state = next_state

            evaluation_scores.append(score)

        env.close()
        return np.mean(evaluation_scores), np.std(evaluation_scores)

    def done(self):
        self.ppo_model.train()
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        log_prob_actions = torch.cat(self.log_prob_actions)
        values = torch.cat(self.values)
        returns = calculate_returns(self.rewards, self.discount_factor)
        advantages = calculate_advantages(returns, values)
        self.train(states, actions, log_prob_actions, advantages, returns)

        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.values = []
        self.rewards = []


def init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    init_seed(7)
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    scores, episodes = [], []
    EPOCHS = 1000
    TARGET_SCORE = 260

    wandb.init(
        project='ppo',
        config={
            'seed': 7,
            'state_size': state_size,
            'action_size': action_size,
            'epochs': EPOCHS,
            'target_score': TARGET_SCORE,
            'dropout_rate': 0.0,

            # 'learning_rate': 0.0005,
            # 'discount_factor': 0.99,
            # 'eps_clip': 0.2,
            # 'k_epoch': 5,
            # 'is_prelu': True,
            # 'is_custom_init': True
        }
    )

    agent = PPOAgent(state_size, action_size, learning_rate=wandb.config.learning_rate,
                     discount_factor=wandb.config.discount_factor, eps_clip=wandb.config.eps_clip,
                     k_epoch=wandb.config.k_epoch, dropout_rate=wandb.config.dropout_rate,
                     is_prelu=wandb.config.is_prelu, is_custom_init=wandb.config.is_custom_init)
    max_score = -9999999999

    for epoch in range(EPOCHS):
        done = False
        truncated = False
        score = 0

        state, _ = env.reset()
        while not (done or truncated):
            action, log_prob_action, value = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action.item())
            agent.append_transition(state, action, log_prob_action, value, reward)

            score += reward
            state = next_state
            if agent.step_counter % 100 == 0:
                wandb.log({'score': score, 'step_counter': agent.step_counter})

            if done or truncated:
                agent.done()

                scores.append(score)
                episodes.append(epoch)
                avg_score = np.mean(scores[-min(30, len(scores)):])
                max_score = max(max_score, avg_score)
                # print(
                #     f'episode:{epoch} '
                #     f'score:{score:.3f}, '
                #     f'avg_score:{avg_score:.3f}, '
                #     f'step_counter:{agent.step_counter}'
                # )
                wandb.log({'avg_score': avg_score})

        if (epoch + 1) % 25 == 0:
            mean_score, std_score = agent.evaluate(num_episodes=5, render=False)
            max_score = max(max_score, mean_score)
            wandb.log({'max_score': max_score})

        if avg_score > TARGET_SCORE:
            print(f'Solved in episode: {epoch + 1}')
            break

    # def plot(scores, episodes):
    #     import matplotlib.pyplot as plt
    #     plt.plot(episodes, scores)
    #     plt.title('original DQN For LunarLander-v3')
    #     plt.xlabel('Episode', fontsize=14)
    #     plt.ylabel('Score', fontsize=14)
    #     plt.grid()
    #     plt.show()
    #
    # plot(scores, episodes)
    mean_score, std_score = agent.evaluate(num_episodes=5, render=False)
    print(f'Evaluated Result(Mean Score: {mean_score:.3f}, Std Score: {std_score:.3f})')
    wandb.log({'mean_score': mean_score, 'std_score': std_score})


if __name__ == '__main__':
    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep3',
        'metric': {'goal': 'maximize', 'name': 'max_score'},
        'parameters':
            {
                'learning_rate': {'distribution': 'log_uniform_values', 'max': 0.001, 'min': 0.00001},
                'discount_factor': {'distribution': 'log_uniform_values', 'max': 1.0, 'min': 0.98},
                'eps_clip': {'distribution': 'log_uniform_values', 'max': 0.3, 'min': 0.01},
                'k_epoch': {'distribution': 'int_uniform', 'max': 10, 'min': 3},
                # 'dropout_rate': {'distribution': 'log_uniform_values', 'max': 0.00, 'min': 0.00001},
                'is_prelu': {'values': [True]},
                'is_custom_init': {'values': [True]}
            }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='ppo')
    wandb.agent(sweep_id=sweep_id, function=main, count=60)
