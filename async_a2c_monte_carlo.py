'''
Main Process 의 AsyncAgent 는 Parameter Server 로 작동
torch.multiprocessing 을 사용하여 global model 과 global optimizer 를 공유
NUM_WORKERS: worker process(async agent) 의 개수
'''

import abc
from typing import Any
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import wandb
import matplotlib.pyplot as plt
from collections import deque
import time

NUM_WORKERS = 4

def init_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gym.utils.seeding.np_random(seed)

class ActorCriticModel(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(ActorCriticModel, self).__init__()
        # Deeper network with more capacity
        self.shared_fc1 = nn.Linear(state_size, 512)
        self.shared_fc2 = nn.Linear(512, 256)
        self.shared_fc3 = nn.Linear(256, 128)
        
        self.actor_fc = nn.Linear(128, action_size)
        self.critic_fc = nn.Linear(128, 1)
        
        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        x = F.relu(self.shared_fc3(x))
        
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        state_value = self.critic_fc(x)
        # Ensure no NaN
        #action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        return action_probs, state_value

class A2CMonteCarloAgent:
    def __init__(self, state_size: int, action_size: int, 
                 discount_factor: float = 0.99,
                 learning_rate: float = 0.001,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.05,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.99,
                 epsilon_min: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters from arguments
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize model and optimizer
        self.model = ActorCriticModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate, 
                                    eps=1e-5)

    def get_action(self, state: np.ndarray) -> int:
        # Convert state to tensor and move to device
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if np.random.random() > self.epsilon:
            # Get action probabilities
            with torch.no_grad():
                action_probs, _ = self.model(state)
                action_probs = action_probs.cpu().numpy().flatten()
            action = np.random.choice(self.action_size, p=action_probs)
            #action = np.argmax(action_probs)  # Greedy action
        else:
            action = np.random.randint(self.action_size)  # Random action
        
        return action

    def train(self, trajectory):
        # Unpack trajectory
        states, actions, rewards, dones = zip(*trajectory)

        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute returns using Monte Carlo estimation
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.discount_factor * G * (1 - done)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward pass
        action_probs, state_values = self.model(states)
        state_values = state_values.squeeze()

        # Compute log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        # Compute advantages
        advantages = returns - state_values.detach()

        # Actor loss with entropy bonus
        actor_loss = -(log_probs * advantages).mean()
        entropy_loss = -(action_probs * torch.log(action_probs + 1e-10)).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(state_values, returns)

        # Total loss
        loss = (actor_loss 
                + self.value_loss_coef * critic_loss 
                - self.entropy_coef * entropy_loss)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # local model 의 optimizer.step() 는 불필요하지만 제거하지 않음. 
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def evaluate(self, env, num_episodes=10):
        epsilon = self.epsilon
        self.epsilon = 0.0
        eval_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state

            eval_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}")
        self.epsilon = epsilon
        return np.mean(eval_rewards), np.std(eval_rewards)
    

    
    
class AsyncAgent:
    def __init__(self, 
                 state_size: int, action_size: int, discount_factor: float = 0.99):
        self.discount_factor = discount_factor
        self.agent = A2CMonteCarloAgent(state_size=state_size, action_size=action_size, discount_factor=discount_factor)
        
    @staticmethod
    def run_worker(worker_id, env_name, global_agent, global_optimizer, lock, res_queue, iQuit):
        env = gym.make(env_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        discount_factor = global_agent.discount_factor
        agent = A2CMonteCarloAgent(state_size=state_size, action_size=action_size, discount_factor=discount_factor)
        agent.model.load_state_dict(global_agent.model.state_dict())
        #agent = deepcopy(global_agent) # local agent
        while iQuit.value == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            trajectory = []
            done = False
            turncated = False
            score = 0
            state, _ = env.reset()
            while not(done or turncated):
                action = agent.get_action(state)
                next_state, reward, done, turncated, info = env.step(action)
                trajectory.append((state, action, reward, done))
                state = next_state
                score += reward
            actor_loss, critic_loss, entropy_loss = agent.train(trajectory)
            
            with lock:
                global_optimizer.zero_grad()
                for local_param, global_param in zip(agent.model.parameters(), global_agent.model.parameters()):
                    global_param._grad = local_param.grad
                global_optimizer.step()
                state_dict = global_agent.model.state_dict()
                agent.model.load_state_dict(state_dict)
                res_queue.put((actor_loss, critic_loss, entropy_loss))
        print(f'End of worker {worker_id}')

        
    def train(self, num_episodes=1000, num_workers=1):
        env_name = 'LunarLander-v3'        
        env = gym.make(env_name)
        global_agent = self.agent
        global_optimizer = self.agent.optimizer
        lock = mp.Lock()
        res_queue = mp.Queue() 
        pr_workers = []
        iQuits = []
        for i in range(num_workers):
            iquit = mp.Value('i', 0)
            pr = mp.Process(target=self.run_worker, args=(i, env_name, global_agent, global_optimizer, lock, res_queue, iquit))
            pr.start()
            pr_workers.append(pr)
            iQuits.append(iquit)
            
            
        count = 0
        t0 = time.time()
        while count < num_episodes:
            actor_loss, critic_loss, entropy_loss = res_queue.get()
            count += 1
            #print(count, actor_loss, critic_loss, entropy_loss)
            if count % 100 == 0:
                mean_reward, std_reward = global_agent.evaluate(env)
                print(f'** eval: count {count}, mean_reward {mean_reward}, std_reward {std_reward}, elapsed {time.time() - t0}')
        
        for i, iquit in enumerate(iQuits):
            iquit.value = 1
        
        
    
def main(num_workers=1):
    mp.set_start_method('spawn')
    env_name = 'LunarLander-v3'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = AsyncAgent(state_size=state_size, action_size=action_size, discount_factor=0.99)
    agent.train(num_episodes=10000, num_workers=num_workers)

if __name__ == "__main__":
    main(NUM_WORKERS)

