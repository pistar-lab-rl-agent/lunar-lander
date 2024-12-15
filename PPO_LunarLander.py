# PPO Algorithm For LunarLander-v3
# It took less than 1 min to solve the env on mac M2max

# Required Libraries
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time

# Set random seed for torch module
torch.manual_seed(17)
np.random.seed(17)

# Global Variables
TARGET_SCORE = 260
EPOCHS = 5000

# Networks
class Actor(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        logits = self.network(x)  # [batch, n_actions]
        return F.softmax(logits, dim=-1)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)

# Agent
class PPOAgent:
    def __init__(self,
                 env,
                 hidden_dim=128,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 lam=0.95,
                 clip_range=0.2,
                 entropy_coef=0.01,
                 vf_coef=0.5,
                 k_epochs=10,
                 batch_size=64,
                 rollout_steps=2048):
        self.env = env
        self.device = torch.device('cpu')  # change to 'cuda' if GPU is available
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.actor = Actor(obs_dim, n_actions, hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_t)
            dist = Categorical(probs)
            action = dist.sample()
            value = self.critic(state_t)
        return action.item(), dist.log_prob(action), value.item()

    def train(self):
        scores, episodes = [], []
        score, episode = 0, 0
        for train_epoch in range(EPOCHS):
            # Collect samples
            states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
            state, _ = self.env.reset()
            for _ in range(self.rollout_steps):
                action, log_prob, value = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                score += reward

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(done or truncated)

                state = next_state
                if done or truncated:
                    episode += 1
                    scores.append(score)
                    episodes.append(episode)

                    avg_score = np.mean(scores[-min(30, len(scores)):])
                    if episode % 10 == 0:
                        print(f'episode:{episode}, score:{score:.3f}, avg_score:{avg_score:.3f}')
                    if avg_score > TARGET_SCORE:
                        # print("The env is solved!")
                        break
                    score = 0
                    state, _ = self.env.reset()

            if avg_score > TARGET_SCORE:
                print("The env is solved!")
                break

            # GAE Calculation
            with torch.no_grad():
                next_state = torch.FloatTensor(next_state).to(self.device)
                next_value = self.critic(next_state).item()
            values = np.append(values, next_value)
            advantages = np.zeros_like(rewards, dtype=np.float32)
            gae = 0
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
                gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
                advantages[i] = gae
            returns = advantages + values[:-1]

            # Convert lists to tensors
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(np.array(actions)).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)

            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update parameters
            dataset_size = len(states)
            for _ in range(self.k_epochs):
                indices = np.arange(dataset_size)
                np.random.shuffle(indices)
                for start in range(0, dataset_size, self.batch_size):
                    end = start + self.batch_size
                    batch_idx = indices[start:end]

                    batch_states = states[batch_idx]
                    batch_actions = actions[batch_idx]
                    batch_old_log_probs = old_log_probs[batch_idx]
                    batch_advantages = advantages[batch_idx]
                    batch_returns = returns[batch_idx]

                    # Forward pass
                    new_probs = self.actor(batch_states)
                    dist = Categorical(new_probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    new_values = self.critic(batch_states).squeeze(-1)

                    # Compute ratios
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)

                    # Clipped surrogate objective
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Critic loss
                    critic_loss = F.mse_loss(new_values, batch_returns)

                    loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
        return episodes, scores

    def evaluate(self, num_episodes=10, render=False):
        env_eval = gym.make('LunarLander-v3', render_mode='human' if render else None)
        evaluation_rewards = []

        for _ in range(num_episodes):
            state, _ = env_eval.reset()
            episode_reward = 0
            done = False
            truncated = False

            while not (done or truncated):
                state_t = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    action_probs = self.actor(state_t)
                action = torch.argmax(action_probs).item()

                state, reward, done, truncated, _ = env_eval.step(action)
                episode_reward += reward

            evaluation_rewards.append(episode_reward)

        env_eval.close()
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

    # Set random seeds for env
    env.reset(seed=7)
    env.action_space.seed(7)

    agent = PPOAgent(
        env=env,
        hidden_dim=128,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        k_epochs=10,
        batch_size=64,
        rollout_steps=2048
    )

    episodes, scores = agent.train()

    t2 = time.time()
    print("Elapsed Time(min):", round((t2 - t1) / 60, 2))

    # Print scores vs episodes
    agent.plot_scores(episodes, scores)

    # Evaluate trained model
    mean_score, std_score = agent.evaluate(num_episodes=5, render=True)
    print(f'Evaluation Result(Mean Score): {mean_score:.2f} ± {std_score:.2f}')