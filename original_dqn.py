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

class DQNModel(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
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
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.001
        self.batch_size = 64
        self.train_start = 640
        self.step_counter = 0
        self.target_update_period = 200

        # Replay Buffer
        self.memory: deque[tuple] = deque(maxlen=10000)

        # Model and Target Model
        self.policy_net   = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Initial Synchronization of Target Model
        self.update_target_model()

        # For logging
        self.loss_history: list[int] = []
    
    def get_action(self, state: np.ndarray) -> int:
        self.step_counter += 1
        if random.random() <= self.epsilon:
            return random.randrange(0, self.action_size - 1)
        else:
            with torch.no_grad():
                return self.policy_net(
                    torch.FloatTensor(state)
                ).argmax().item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_net.state_dict())

    def train(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([x[STATE] for x in mini_batch]))    # [64,8]
        actions = torch.LongTensor(np.array([x[ACTION] for x in mini_batch]))   # [64]
        rewards = torch.FloatTensor(np.array([x[REWARD] for x in mini_batch]))  # [64]
        next_states = torch.FloatTensor(np.array([x[NEXT_STATE] for x in mini_batch]))  # [64,8]
        dones = torch.FloatTensor(np.array([x[DONE] for x in mini_batch]))      # [64]

        if states.shape != (self.batch_size, self.state_size):
            raise ValueError(
                f"Expected states to have shape {(self.batch_size, self.state_size)}, but got {states.shape}"
            )

        # Current Q Values: Q(s,a)
        curr_Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # [64]

        # Next Q Values: max_a' Q'(s',a')
        next_Q = self.target_model(next_states).squeeze().max(1)[0]  # [64]
        expected_Q = rewards + self.discount_factor * next_Q * (1 - dones)  # [64]

        # Loss
        loss = F.mse_loss(curr_Q, expected_Q.detach())

        # Update Parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log Loss
        self.loss_history.append(loss.item())

        if self.step_counter % self.target_update_period == 0:
            self.update_target_model()

        # Decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def evaluate(self, num_episodes=5, render=False):
        env = gym.make('LunarLander-v3', render_mode='human' if render else None)
        self.policy_net.eval()
        self.epsilon = 0
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
            if done or turncated:
                agent.done()
                
                scores.append(score)
                episodes.append(epoch)
                avg_score = np.mean(scores[-min(30, len(scores)):])
                print(
                    f'episode:{epoch} '
                    f'score:{score:.3f}, '
                    f'avg_score:{avg_score:.3f}, '
                    f'epsilon:{agent.epsilon:.3f}, '
                    f'step_counter:{agent.step_counter}'
                )
    
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
    
if __name__ == "__main__":
    main()
        
        
    
