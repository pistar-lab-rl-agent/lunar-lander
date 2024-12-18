import abc
from collections import deque
import random
from typing import Any
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import math

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

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
    
class DQNModel(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = NoisyLinear(128, 128)
        
        self.a_network = NoisyLinear(128, action_size) # a_network: A(s, a) - 상태의 advantage를 추정
        self.v_network = NoisyLinear(128,1) # v_network: V(s) - 상태의 가치를 추정
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        v = self.v_network(x)
        a = self.a_network(x)
        
        q = v + a - a.mean(-1, keepdim=True).expand_as(a) # (q = v + a) == (q = v + a - mean(a))
        return q
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.fc3.reset_noise()
        self.a_network.reset_noise()
        self.v_network.reset_noise()
        
class DQNAgent:
    def __init__(
        self, 
        state_size: int, 
        action_size: int
    ):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.train_start = 640
        self.step_counter = 0
        self.target_update_period = 200

        # Replay Buffer
        self.memory: deque[tuple] = deque(maxlen=10000)

        # Model and Target Model
        self.policy_net = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Initial Synchronization of Target Model
        self.update_target_model()

        # For logging
        self.loss_history: list[int] = []
        
        # Training mode flag
        self.training = True
    
    def get_action(self, state: np.ndarray) -> int:
        """NoisyNet doesn't need epsilon-greedy since noise is added in the network"""
        self.step_counter += 1
        with torch.no_grad():
            if self.training:
                # During training, use noisy network
                action = self.policy_net(
                    torch.FloatTensor(state)
                ).argmax().item()
            else:
                # During evaluation, use deterministic network (no noise)
                action = self.policy_net(
                    torch.FloatTensor(state)
                ).argmax().item()
        return action
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_net.state_dict())
        
    def train(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([x[STATE] for x in mini_batch]))
        actions = torch.LongTensor(np.array([x[ACTION] for x in mini_batch]))
        rewards = torch.FloatTensor(np.array([x[REWARD] for x in mini_batch]))
        next_states = torch.FloatTensor(np.array([x[NEXT_STATE] for x in mini_batch]))
        dones = torch.FloatTensor(np.array([x[DONE] for x in mini_batch]))

        if states.shape != (self.batch_size, self.state_size):
            raise ValueError(
                f"Expected states to have shape {(self.batch_size, self.state_size)}, but got {states.shape}"
            )

        # Reset noise for both networks
        self.policy_net.reset_noise()
        self.target_model.reset_noise()

        # Current Q-values for the chosen actions
        curr_Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Action selection using the policy network
        MAX_ARGS_IDX = 1
        next_action = self.policy_net(next_states).max(1)[MAX_ARGS_IDX].unsqueeze(1)
        # Q-value evaluation using the target network
        next_Q = self.target_model(next_states).gather(1, next_action).squeeze(1).detach()
        # Target Q-values
        expected_Q = rewards + self.discount_factor * next_Q * (1 - dones)

        # Loss
        loss = F.mse_loss(curr_Q, expected_Q.detach())

        
        # Update Parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy_net.reset_noise()  # Reset noise for each action selection
        self.target_model.reset_noise()  # Reset noise for each action selection

        # Log Loss
        self.loss_history.append(loss.item())

        if self.step_counter % self.target_update_period == 0:
            self.update_target_model()
            
    def evaluate(self, num_episodes=5, render=False):
        env = gym.make('LunarLander-v3', render_mode='human' if render else None)
        self.policy_net.eval()
        self.training = False  # Disable noise during evaluation
        evaluation_scores = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            score = 0
            done = False
            truncated = False

            while not (done or truncated):
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                score += reward
                state = next_state

            evaluation_scores.append(score)

        self.training = True  # Re-enable noise for training
        self.policy_net.train()
        env.close()
        return np.mean(evaluation_scores), np.std(evaluation_scores)
    
    def done(self):
        self.update_target_model()
            

def init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    init_seed(7)
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    EPOCHS = 1000
    TARGET_SCORE = 260

    wandb.init(
        project="pistar-lab-lunar-lander",
        config={
            "seed": 7,
            "state_size": state_size,
            "action_size": action_size,
            "discount_factor": agent.discount_factor,
            "learning_rate": agent.learning_rate,
            "batch_size": agent.batch_size,
            "train_start": agent.train_start,
            "target_update_period": agent.target_update_period,
            "epochs": EPOCHS,
            "target_score": TARGET_SCORE
        }
    )

    for epoch in range(EPOCHS):
        done = False
        turncated = False
        score = 0

        state, _ = env.reset()
        while not(done or turncated):
            action = agent.get_action(state)
            next_state, reward, done, turncated, info = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            if len(agent.memory) >= agent.train_start:
                agent.train()
            score += reward
            state = next_state

            if agent.step_counter % 100 == 0:
                wandb.log({"score": score, "step_counter": agent.step_counter})

            if done or turncated:
                agent.done()
                
                scores.append(score)
                episodes.append(epoch)
                avg_score = np.mean(scores[-min(30, len(scores)):])
                print(
                    f'episode:{epoch} '
                    f'score:{score:.3f}, '
                    f'avg_score:{avg_score:.3f}, '
                    f'step_counter:{agent.step_counter}'
                )
                wandb.log({"avg_score": avg_score})
    
        if avg_score > TARGET_SCORE:
            print(f"Solved in episode: {epoch + 1}")
            break
    
    def plot(scores, episodes):
        import matplotlib.pyplot as plt
        plt.plot(episodes, scores)
        plt.title('original DQN For LunarLander-v3')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid()
        plt.show()

    plot(scores, episodes)
    mean_score, std_score = agent.evaluate(num_episodes=5, render=True)
    print(f"Evaluated Result(Mean Score: {mean_score:.3f}, Std Score: {std_score:.3f})")
    wandb.log({"mean_score": mean_score, "std_score": std_score})
    
if __name__ == "__main__":
    main()
        
        
    