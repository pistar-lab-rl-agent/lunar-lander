'''
Main Process 의 AsyncAgent 는 Parameter Server 로 작동
torch.multiprocessing 을 사용하여 global model 과 global optimizer 를 공유
NUM_WORKERS: worker process(async agent) 의 개수
'''

import abc
from typing import Any
import gymnasium as gym
import numpy as np
import random
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
            return random.randrange(0, self.action_size)
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


        # Current Q-values for the chosen actions
        curr_Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Action selection using the policy network
        MAX_ARGS_IDX = 1
        next_action = self.policy_net(next_states).max(1)[MAX_ARGS_IDX].unsqueeze(1)
        # Q-value evaluation using the target network
        next_Q = self.target_model(next_states).gather(1, next_action).squeeze(1).detach()
        # Target Q-values
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
        return loss.detach().item()

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

    
    
class AsyncAgent:
    def __init__(self, 
                 state_size: int, action_size: int):
        self.agent = DQNAgent(state_size=state_size, action_size=action_size)
        
    @staticmethod
    def run_worker(worker_id, env_name, global_agent, global_optimizer, lock, res_queue, iQuit):
        print(f'worker {worker_id} has started')
        env = gym.make(env_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        agent.policy_net.load_state_dict(global_agent.policy_net.state_dict())
        agent.target_model.load_state_dict(global_agent.target_model.state_dict())
        #agent = deepcopy(global_agent) # local agent
        while iQuit.value == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            done = False
            turncated = False
            score = 0
            state, _ = env.reset()
            while not(done or turncated):
                agent.done()
                action = agent.get_action(state)
                next_state, reward, done, turncated, info = env.step(action)
                agent.memory.append((state, action, reward, next_state, done))
                state = next_state
                score += reward
                if len(agent.memory) >= agent.train_start:
                    loss = agent.train()
                    #print(f'train done')
            
                with lock:
                    #print('with lock')
                    global_optimizer.zero_grad()
                    for local_param, global_param in zip(agent.policy_net.parameters(), global_agent.policy_net.parameters()):
                        global_param._grad = local_param.grad
                    #for local_param, global_param in zip(agent.target_model.parameters(), global_agent.target_model.parameters()):
                    #    global_param._grad = local_param.grad
                    global_optimizer.step()
                    agent.policy_net.load_state_dict(global_agent.policy_net.state_dict())
            res_queue.put(score)
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
            score = res_queue.get()
            count += 1
            
            #print(count, actor_loss, critic_loss, entropy_loss)
            if count % 100 == 0:
                mean_reward, std_reward = global_agent.evaluate()
                print(f'** eval: count {count}, mean_reward {mean_reward}, std_reward {std_reward}, elapsed {time.time() - t0}')
        
        for i, iquit in enumerate(iQuits):
            iquit.value = 1
        
        
    
def main(num_workers=1):
    mp.set_start_method('spawn')
    env_name = 'LunarLander-v3'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = AsyncAgent(state_size=state_size, action_size=action_size)
    agent.train(num_episodes=10000, num_workers=num_workers)

if __name__ == "__main__":
    main(NUM_WORKERS)

