# A2C Algorithm For LunarLander-v3
# Monte Carlo method is used
# It took about 16 min to solve the env on mac M2max

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import time

# Set random seed for torch module
torch.manual_seed(15)

class Actor(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)


class A2CMCAgent:
    def __init__(self, hidden_dim=128, actor_lr=0.001, critic_lr=0.001, gamma=0.99, entropy_coef=0.01):
        self.device = torch.device('cpu')

        # Initialize networks
        self.actor = Actor(
            input_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.critic = Critic(
            input_dim=env.observation_space.shape[0],
            hidden_dim=hidden_dim
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)  # [8]
        action_probs = self.actor(state)  # [4]
        dist = Categorical(action_probs)
        action = dist.sample()  # []
        state_value = self.critic(state)  # []

        return action.item(), dist.log_prob(action), state_value

    def evaluate(self, num_episodes=10, render=False):
        env = gym.make('LunarLander-v3', render_mode='human' if render else None)
        evaluation_rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False

            while not (done or truncated):
                state = torch.FloatTensor(state).to(self.device)
                action_probs = self.actor(state)
                action = torch.argmax(action_probs).item()

                state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward

            evaluation_rewards.append(episode_reward)

        env.close()
        return np.mean(evaluation_rewards), np.std(evaluation_rewards)

    def plot_scores(self, episodes, scores):
        plt.plot(episodes, scores)
        plt.title('original DQN For LunarLander-v3')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    t1 = time.time()
    env = gym.make('LunarLander-v3')

    # Set random seed for env
    env.reset(seed=12)
    env.action_space.seed(12)

    # Initialize agent
    agent = A2CMCAgent(
        hidden_dim=128,
        actor_lr=0.0002,
        critic_lr=0.0002,
        gamma=0.99,
        entropy_coef=0.001
    )

    scores, episodes = [], []
    EPOCHS = 10000
    TARGET_SCORE = 260

    for episode in range(1, EPOCHS + 1):

        done = False
        truncated = False
        score = 0
        step_counter = 0  # max=1000 -> done=True

        values = []
        log_probs = []
        rewards = []

        state, _ = env.reset()

        while not (done or truncated):
            # determine action in state
            action, log_prob, value = agent.get_action(state)

            # transition to next step by env
            next_state, reward, done, truncated, _ = env.step(action)
            step_counter += 1

            # Store step information
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            score += reward

            state = next_state

        # Record score and print out results
        scores.append(score)
        episodes.append(episode)

        avg_score = np.mean(scores[-min(30, len(scores)):])
        if episode % 100 == 0:
            print(f'episode:{episode}, score:{score:.3f}, avg_score:{avg_score:.3f}, step_counter:{step_counter}')

        if avg_score > TARGET_SCORE:
            #         torch.save(agent.model.state_dict(), "./save_model/lunarlander_v3_dqn_trained_01.pth")
            #                 np.savez_compressed('perf_rec_01.npz', np.array(scores))
            t2 = time.time()
            print("The env is solved!")
            print("Elapsed Time(min):", round((t2 - t1) / 60, 2))
            break

        # Convert lists to tensors using torch.stack
        values = torch.stack(values).squeeze(-1)
        log_probs = torch.stack(log_probs)

        # Calculate returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + agent.gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        returns = torch.FloatTensor(returns).to(agent.device)
        advantages = returns - values

        # Update critic
        critic_loss = advantages.pow(2).mean()
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Update actor
        actor_loss = -(log_probs * advantages.detach()).mean()
        entropy = -(log_probs.exp() * log_probs).mean()  # Entropy bonus
        actor_total_loss = actor_loss - agent.entropy_coef * entropy

        agent.actor_optimizer.zero_grad()
        actor_total_loss.backward()
        agent.actor_optimizer.step()

    if avg_score < TARGET_SCORE:
        #         torch.save(agent.model.state_dict(), "./save_model/lunarlander_v3_dqn_trained_01.pth")
        #                 np.savez_compressed('perf_rec_01.npz', np.array(scores))
        t2 = time.time()
        print("The env is not solved!")
        print("Elapsed Time(min):", round((t2 - t1) / 60, 2))

    # Print scores vs episodes
    agent.plot_scores(episodes, scores)

    # Evaluate trained model
    mean_score, std_score = agent.evaluate(num_episodes=5, render=True)
    print(f'Evaluation Result(Mean Score): {mean_score:.2f} ± {std_score:.2f}')
