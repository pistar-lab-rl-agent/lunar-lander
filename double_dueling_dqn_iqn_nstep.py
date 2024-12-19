import abc
from collections import deque
import random
from typing import Any, Optional, Tuple, Dict
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
        
    def sample_batch(self, batch_size: int):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        
        batch = []
        for idx in indices:
            batch.append([
                self.state_buf[idx],
                self.action_buf[idx],
                self.reward_buf[idx],
                self.next_state_buf[idx],
                self.done_buf[idx]
            ])
        
        return batch
        
    def sample_minibatch(self, batch_size: int) -> Dict[str, np.ndarray]:
        # 현재 버퍼에 있는 데이터 크기 내에서 랜덤하게 인덱스 선택
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return dict(
            states=self.state_buf[indices],
            next_states=self.next_state_buf[indices],
            actions=self.action_buf[indices],
            rewards=self.reward_buf[indices],
            dones=self.done_buf[indices]
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
    
class IQNModel(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(IQNModel, self).__init__()
        # IQN specific parameters
        self.n_cos = 64
        self.n_tau = 8
        self.K = 32  # number of samples for inference
        
        # Cosine embedding
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos)
        
        # Feature network
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        
        # Cosine embedding network
        self.cos_embedding = nn.Linear(self.n_cos, 128)
        
        # Dueling networks
        self.v_network = nn.Linear(128, 1)  # State value
        self.a_network = nn.Linear(128, action_size)  # Advantage for each action

    def calc_cos(self, batch_size: int, n_tau: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate cosine values based on number of tau samples."""
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1)
        if self.pis.device != taus.device:
            self.pis = self.pis.to(taus.device)
        cos = torch.cos(taus * self.pis)
        return cos, taus

    def forward(self, x: torch.Tensor, num_tau: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # State embedding
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # (batch_size, layer_size)
        
        # Cosine embedding
        cos, taus = self.calc_cos(batch_size, num_tau)  # cos: (batch_size, num_tau, n_cos)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = F.relu(self.cos_embedding(cos)).view(batch_size, num_tau, -1)
        
        # Combine embeddings
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, -1)
        
        # Dueling architecture
        v = self.v_network(x)  # State value
        a = self.a_network(x)  # Advantage for each action
        
        # Combine value and advantage
        v = v.view(batch_size, num_tau, 1)
        a = a.view(batch_size, num_tau, -1)
        
        # Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=2, keepdim=True))
        
        return q, taus

    def get_qvalues(self, x: torch.Tensor) -> torch.Tensor:
        """Get Q-values for inference."""
        quantiles, _ = self.forward(x, self.K)
        return quantiles.mean(dim=1)  # Average across tau samples

STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
DONE = 4

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

        # Replay Buffer
        self.n_step = n_step
        self.memory = ReplayBuffer(
            state_size=state_size,
            max_size=10000,
            batch_size=self.batch_size,
            n_step=n_step,
            gamma=self.discount_factor
        )

        # Model and Target Model
        self.policy_net = IQNModel(state_size, action_size)
        self.target_model = IQNModel(state_size, action_size)
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
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net.get_qvalues(state)
                return q_values.argmax().item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_net.state_dict())

    def calculate_huber_loss(self, td_errors: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        """Calculate huber loss element-wise."""
        return torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa)
        )

    def train(self):
        mini_batch = self.memory.sample_batch(self.batch_size)

        states = torch.FloatTensor(np.array([x[STATE] for x in mini_batch]))
        actions = torch.LongTensor(np.array([x[ACTION] for x in mini_batch]))
        rewards = torch.FloatTensor(np.array([x[REWARD] for x in mini_batch]))
        next_states = torch.FloatTensor(np.array([x[NEXT_STATE] for x in mini_batch]))
        dones = torch.FloatTensor(np.array([x[DONE] for x in mini_batch]))

        # Double DQN: Use policy network to select actions
        with torch.no_grad():
            next_actions = self.policy_net.get_qvalues(next_states).argmax(dim=1)
            next_Q, _ = self.target_model(next_states)
            next_Q = next_Q.gather(2, next_actions.view(-1, 1, 1).expand(-1, self.policy_net.n_tau, 1))
            target_Q = rewards.unsqueeze(-1).unsqueeze(-1) + \
                      (self.discount_factor * next_Q * (1 - dones.unsqueeze(-1).unsqueeze(-1)))

        # Current Q-values
        current_Q, taus = self.policy_net(states)
        current_Q = current_Q.gather(2, actions.unsqueeze(-1).unsqueeze(1).expand(-1, self.policy_net.n_tau, 1))

        # Calculate quantile huber loss
        td_errors = target_Q - current_Q
        huber_loss = self.calculate_huber_loss(td_errors)
        quantile_loss = torch.abs(taus - (td_errors.detach() < 0).float()) * huber_loss

        loss = quantile_loss.sum(dim=1).mean(dim=1).mean()

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log loss
        self.loss_history.append(loss.item())

        # Update target network if needed
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
        
        
    