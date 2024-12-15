import abc
import random
from collections import deque
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

# Seed Initialization
def init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Base Agent Class
class AbsAgent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
DONE = 4

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_size: int, action_size: int, max_action: float):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.max_action = max_action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DDPG Agent
class DDPGAgent(AbsAgent):
    def __init__(self, state_size: int, action_size: int, max_action: float):
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action

        # Hyperparameters
        self.discount_factor = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.train_start = 1000

        # Replay Buffer
        self.memory: deque[tuple] = deque(maxlen=100000)

        # Actor and Critic Networks
        self.actor = Actor(state_size, action_size, max_action)
        self.target_actor = Actor(state_size, action_size, max_action)
        self.critic = Critic(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        # Initialize Target Networks
        self.update_target_networks(tau=1.0)

        # Noise
        self.noise = np.zeros(self.action_size)
        self.noise_std = 0.2

    def get_action(self, state: np.ndarray, noise=True) -> np.ndarray:
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        if noise:
            action += np.random.normal(0, self.noise_std, size=self.action_size)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self):
        if len(self.memory) < self.train_start:
            return

        # Sample mini-batch
        mini_batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([x[STATE] for x in mini_batch]))
        actions = torch.FloatTensor(np.array([x[ACTION] for x in mini_batch]))
        rewards = torch.FloatTensor(np.array([x[REWARD] for x in mini_batch]))
        next_states = torch.FloatTensor(np.array([x[NEXT_STATE] for x in mini_batch]))
        dones = torch.FloatTensor(np.array([x[DONE] for x in mini_batch]))

        # Critic Loss
        next_actions = self.target_actor(next_states)
        next_Q = self.target_critic(next_states, next_actions).squeeze(1)
        target_Q = rewards + self.discount_factor * next_Q * (1 - dones)
        curr_Q = self.critic(states, actions).squeeze(1)
        critic_loss = F.mse_loss(curr_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        self.update_target_networks()

    def evaluate(self, num_episodes=5, render=False):
        env = gym.make('LunarLanderContinuous-v2', render_mode='human' if render else None)
        self.actor.eval()
        scores = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            score = 0
            done = False

            while not done:
                action = self.get_action(state, noise=False)
                next_state, reward, done, _, _ = env.step(action)
                score += reward
                state = next_state

            scores.append(score)

        env.close()
        return np.mean(scores), np.std(scores)

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def remember(self, *args):
        self.memory.append(args)

# Main Function
def main():
    init_seed(7)
    env = gym.make('LunarLanderContinuous-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_size, action_size, max_action)

    scores, episodes = [], []
    EPOCHS = 1000
    TARGET_SCORE = 200

    wandb.init(project="ddpg-lunar-lander", config={"epochs": EPOCHS, "target_score": TARGET_SCORE})

    for epoch in range(EPOCHS):
        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            score += reward
            state = next_state

        scores.append(score)
        episodes.append(epoch)
        avg_score = np.mean(scores[-min(30, len(scores)):])

        print(f'Episode: {epoch}, Score: {score:.2f}, Avg Score: {avg_score:.2f}')
        wandb.log({"score": score, "avg_score": avg_score})

        if avg_score >= TARGET_SCORE:
            print(f"Solved in Episode: {epoch + 1}")
            break

    mean_score, std_score = agent.evaluate(num_episodes=5, render=True)
    print(f"Evaluation Result (Mean Score: {mean_score:.2f}, Std Score: {std_score:.2f})")
    wandb.log({"mean_score": mean_score, "std_score": std_score})

if __name__ == "__main__":
    main()
