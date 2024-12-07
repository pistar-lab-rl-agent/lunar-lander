import abc
import random
from typing import Any
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import time
import wandb

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

def init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class AbsAgent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        pass

    @abc.abstractmethod
    def train(self, *arge, **kwargs) -> Any: 
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        pass

    @abc.abstractmethod
    def done(self, *args, **kwargs) -> Any:
        pass

STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
DONE = 4


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
        return F.softmax(self.network(x), dim=-1)


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


class A2CNStepAgent(AbsAgent):
    def __init__(
        self, 
        state_size :int,
        action_size: int,
        hidden_dim=128, 
        actor_lr=0.002, 
        critic_lr=0.002, 
        gamma=0.99,
        entropy_coef=0.01, 
        n_steps=10
    ):
        self.n_steps = n_steps

        self.actor = Actor(
            input_dim=state_size,
            n_actions=action_size,
            hidden_dim=hidden_dim
        ).to(device)

        self.critic = Critic(
            input_dim=state_size,
            hidden_dim=hidden_dim
        ).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Memory buffer for n-step returns
        self.memory: deque[tuple] = deque(maxlen=n_steps)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)     # [8]
        action_probs = self.actor(state)                # [4]
        dist = Categorical(action_probs)
        action = dist.sample()                          # []
        state_value = self.critic(state)                # []
        return action.item(), dist.log_prob(action), state_value, state
    
    def train(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        # Get action info for current state
        action, log_prob, value, state_tensor = self.get_action(state)
        self.memory.append((action, log_prob, reward, state_tensor, value))
        
        # Update if memory is full or episode ends
        if len(self.memory) == self.n_steps or done:
            # Get next state value for bootstrapping
            with torch.no_grad():
                next_state_value = 0 if done else self.critic(
                    torch.FloatTensor(next_state).to(device)
                ).item()

            # Compute n-step returns and get relevant tensors
            returns, states, log_probs = self.compute_nstep_returns(next_state_value)
            values = self.critic(states).squeeze(-1)
            advantages = returns - values

            # Update critic
            critic_loss = advantages.pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actor
            actor_loss = -(log_probs * advantages.detach()).mean()
            entropy = -(log_probs.exp() * log_probs).mean()
            actor_total_loss = actor_loss - self.entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            actor_total_loss.backward()
            self.actor_optimizer.step()

            # Clear memory after update
            self.memory.clear()

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
    
    def done(self): ...

    def compute_nstep_returns(self, next_state_value=0):
        rewards = [transition[REWARD] for transition in self.memory]    # n floats
        states = [transition[NEXT_STATE] for transition in self.memory] # n [8] tensors
        log_probs = [transition[ACTION] for transition in self.memory]  # n [] tensors

        returns = []
        R = next_state_value

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(device)    # [n]
        states = torch.stack(states)                            # [n,8]
        log_probs = torch.stack(log_probs)                      # [n]
        return returns, states, log_probs

def main():
    init_seed(7)
    t1 = time.time()
    env = gym.make('LunarLander-v3')
    env.reset(seed=7)
    env.action_space.seed(7)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_dim = 128
    actor_lr = 0.0003
    critic_lr = 0.0002
    gamma = 0.999
    entropy_coef = 0.001
    n_steps = 9

    agent = A2CNStepAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_dim=hidden_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        entropy_coef=entropy_coef,
        n_steps=n_steps
    )

    # Training parameters
    scores, episodes = [], []
    EPOCHS = 7000
    TARGET_SCORE = 260

    wandb.init(
        project="pistar-lab-lunar-lander",
        config={
            "seed": 7,
            "state_size": state_size,
            "action_size": action_size,
            "hidden_dim": hidden_dim,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "gamma": gamma,
            "entropy_coef": entropy_coef,
            "n_steps": n_steps,
            "epochs": EPOCHS,
            "target_score": TARGET_SCORE
        }
    )

    global_step_counter = 0
    for episode in range(1, EPOCHS + 1):
        done = False
        truncated = False
        score = 0
        step_counter = 0
        agent.memory.clear()  # Clear memory at start of episode
        state, _ = env.reset()

        while not (done or truncated):
            step_counter += 1
            global_step_counter += 1
            
            action, _, _, _ = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.train(state, action, reward, next_state, done or truncated)

            if global_step_counter % 100 == 0:
                wandb.log({"score": score, "step_counter": step_counter, "global_step_counter": global_step_counter})
            
            score += reward
            state = next_state
            
            if done or truncated:
                agent.done()

        # Record score and print results
        scores.append(score)
        episodes.append(episode)
        avg_score = np.mean(scores[-min(30, len(scores)):])
        print(
            f'episode:{episode}, '
            f'score:{score:.3f}, '
            f'avg_score:{avg_score:.3f}, '
            f'step_counter:{step_counter}, '
            f'global_step_counter:{global_step_counter}'
        )
        wandb.log({"avg_score": avg_score})

        if avg_score > TARGET_SCORE:
            t2 = time.time()
            print("The env is solved!")
            print("Elapsed Time(min):", round((t2 - t1) / 60, 2))
            wandb.log({"elapsed_time": round((t2 - t1) / 60, 2)})
            break
            
    def plot(scores, episodes):
        import matplotlib.pyplot as plt
        plt.plot(episodes, scores)
        plt.title('original DQN For LunarLander-v3')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid()
        plt.show()

    # Evaluate trained model
    plot(scores, episodes)
    mean_score, std_score = agent.evaluate(num_episodes=5, render=True)
    print(f"Evaluated Result(Mean Score: {mean_score:.3f}, Std Score: {std_score:.3f})")
    wandb.log({"mean_score": mean_score, "std_score": std_score})

if __name__ == "__main__":
    main()