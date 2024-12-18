import abc
from collections import deque
from typing import Any, Dict, Tuple
import random
from typing import Any
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

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


class ReplayBuffer:
    def __init__(
        self, 
        state_size: int,
        max_size: int = 10000,
        batch_size: int = 64,
        n_step: int = 3,
        gamma: float = 0.99
    ):
        self.state_buf = np.zeros([max_size, state_size], dtype=np.float32)
        self.next_state_buf = np.zeros([max_size, state_size], dtype=np.float32)
        self.action_buf = np.zeros([max_size], dtype=np.float32)
        self.reward_buf = np.zeros([max_size], dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr, self.size = 0, 0
        
        # N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
        
    def store(
        self, 
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) < self.n_step:
            return ()
            
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]
        
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]
        
    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        
        return dict(
            states=self.state_buf[idxs],
            next_states=self.next_state_buf[idxs],
            actions=self.action_buf[idxs],
            rewards=self.reward_buf[idxs],
            dones=self.done_buf[idxs],
            indices=idxs,
        )
        
    def _get_n_step_info(self) -> Tuple[float, np.ndarray, bool]:
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
            
        return reward, next_state, done
        
    def __len__(self) -> int:
        return self.size
    
    
    
class DQNModel(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        
        self.a_network = nn.Linear(128, action_size) # a_network: A(s, a) - 상태의 advantage를 추정
        self.v_network = nn.Linear(128,1) # v_network: V(s) - 상태의 가치를 추정
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        v = self.v_network(x)
        a = self.a_network(x)
        
        q = v + a - a.mean(-1, keepdim=True).expand_as(a) # (q = v + a) == (q = v + a - mean(a))
        return q
    
class DQNAgent:
    def __init__(
        self, 
        state_size: int, 
        action_size: int,
        n_step: int = 3  # N-step parameter
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
        
        # N-step Learning
        self.n_step = n_step
        self.memory = ReplayBuffer(
            state_size=state_size,
            max_size=10000,
            batch_size=self.batch_size,
            n_step=n_step,
            gamma=self.discount_factor
        )

        # Model and Target Model
        self.policy_net = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Initial Synchronization of Target Model
        self.update_target_model()

        # For logging
        self.loss_history: list[float] = []
        
            
    def get_action(self, state: np.ndarray) -> int:
        self.step_counter += 1
        if random.random() <= self.epsilon:
            return random.randrange(0, self.action_size)
        else:
            with torch.no_grad():
                return self.policy_net(
                    torch.FloatTensor(state)
                ).argmax().item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_net.state_dict())

    def train(self):
        samples = self.memory.sample_batch()
        
        states = torch.FloatTensor(samples["states"])
        actions = torch.LongTensor(samples["actions"])
        rewards = torch.FloatTensor(samples["rewards"])
        next_states = torch.FloatTensor(samples["next_states"])
        dones = torch.FloatTensor(samples["dones"])

        # Current Q-values for the chosen actions
        curr_Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Action selection using the policy network
        next_action = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        
        # Q-value evaluation using the target network
        next_Q = self.target_model(next_states).gather(1, next_action).squeeze(1).detach()
        
        # N-step returns
        gamma_n = self.discount_factor ** self.n_step
        expected_Q = rewards + gamma_n * next_Q * (1 - dones)

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

    wandb.init(
        project="pistar-lab-lunar-lander",
        config={
            "seed": 7,
            "state_size": state_size,
            "action_size": action_size,
            "discount_factor": agent.discount_factor,
            "learning_rate": agent.learning_rate,
            "epsilon": agent.epsilon,
            "epsilon_decay": agent.epsilon_decay,
            "epsilon_min": agent.epsilon_min,
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
            agent.memory.store(state, action, reward, next_state, done)
            if len(agent.memory) >= agent.train_start:
                agent.train()
            score += reward
            state = next_state

            if agent.step_counter % 100 == 0:
                wandb.log({"score": score, "epsilon": agent.epsilon, "step_counter": agent.step_counter})

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
        
        
    