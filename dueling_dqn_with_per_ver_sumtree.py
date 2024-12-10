# It took about 22 min to solve the env at 1560 epochs on RTX 3080 Ti Laptop, i9-12900H

import abc
from collections import deque
import random
from typing import Any, Callable, Dict, List
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import operator # Segment Tree

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

##### SegmentTree taken from github repo: #####
# https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/segment_tree.py ###

"""Segment tree for Prioritized Replay Buffer."""
class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity # convert array idx to tree idx
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            # set parent's value (operation is add or min)
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity # convert tree idx to arrays idx


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)

##### End of Segment Tree ######

class ReplayBuffer:
    """A simple numpy replay buffer."""
    """ Taken from https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb """

    def __init__(self, state_size: int, maxlen: int, batch_size: int = 64):
        self.state_buf = np.zeros([maxlen, state_size], dtype=np.float32)
        self.next_state_buf = np.zeros([maxlen, state_size], dtype=np.float32)
        self.acts_buf = np.zeros([maxlen], dtype=np.float32)
        self.rews_buf = np.zeros([maxlen], dtype=np.float32)
        self.done_buf = np.zeros(maxlen, dtype=np.float32)
        self.max_size, self.batch_size = maxlen, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        state: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_state: np.ndarray, 
        done: bool,
    ):
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(state=self.state_buf[idxs],
                    next_state=self.next_state_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer."""
    """ Reference: https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb """

    def __init__(self, state_size: int, maxlen: int, batch_size: int, alpha: float=0.6, beta: float=0.4) -> None:
        super(PrioritizedReplayBuffer, self).__init__(state_size=state_size, maxlen=maxlen, batch_size=batch_size)
        self.maxlen = maxlen
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.max_priority = 1.0    # pi = 1/rank(i) or |delta_i| + eps
        self.tree_ptr = 0
        
        # must be power of 2:
        # https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
        capacity = 1 
        while capacity < maxlen:
            capacity *= 2

        self.sum_tree = SumSegmentTree(capacity) # to get P(i)
        self.min_tree = MinSegmentTree(capacity) # to find max weight to normalize
    
    def store(self, 
        state: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_state: np.ndarray, 
        done: bool,              
              ):
        super().store(state, act, rew, next_state, done)
        
        # Give highest priority for new experience
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.maxlen
    
    
    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        
        indices = self._sample_proportional()
        
        state = self.state_buf[indices]
        next_state = self.next_state_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, self.beta) for i in indices])
        
        return dict(
            state=state,
            next_state=next_state,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices, # type: ignore
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            
            # idx for array

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight for importance sampling
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

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
        v = self.v_network(x) # [bsz, 1]
        a = self.a_network(x) # [bsz, 4]
        
        q = v + a - a.mean(-1, keepdim=True).expand_as(a) # (q = v + a) == (q = v + a - mean(a))
        return q # [bsz, 4]
    
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
        self.learning_rate = 0.00010
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.001
        self.batch_size = 64
        self.train_start = 64 # 640
        self.step_counter = 0
        self.target_update_period = 200
        # PER params
        self.replay_period = 300 # prioritized replay period
        self.per_beta_slope = 1.2e-6
        self.prior_eps = 1e-6

        # Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(state_size=self.state_size, maxlen=10000, batch_size=self.batch_size)
        # self.memory = ReplayBuffer(state_size=self.state_size, maxlen=10000, batch_size=self.batch_size)
        # self.memory: deque[tuple] = deque(maxlen=10000)

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # Model and Target Model
        self.policy_net   = DQNModel(state_size, action_size).to(self.device)
        self.target_model = DQNModel(state_size, action_size).to(self.device)
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
                    torch.FloatTensor(state).to(self.device)
                ).argmax().cpu().item()

    
    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_net.state_dict())

    def train(self):
        mini_batch = self.memory.sample_batch()
        self.memory.beta = min(1.0, self.memory.beta + self.per_beta_slope) # update beta

        states = torch.FloatTensor(mini_batch["state"]).to(self.device)    # [64,8]
        actions = torch.LongTensor(mini_batch["acts"]).to(self.device)  # [64]
        rewards = torch.FloatTensor(mini_batch["rews"]).to(self.device)  # [64]
        next_states = torch.FloatTensor(mini_batch["next_state"]).to(self.device)  # [64,8]
        dones = torch.FloatTensor(mini_batch["done"]).to(self.device)      # [64]

        # PER        
        weights = torch.FloatTensor(mini_batch["weights"]).to(self.device)
        indices = mini_batch["indices"]

        if states.shape != (self.batch_size, self.state_size):
            raise ValueError(
                f"Expected states to have shape {(self.batch_size, self.state_size)}, but got {states.shape}"
            )

        
        # Current Q Values: Q(s,a)
        curr_Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # [64]

        # Next Q Values: max_a' Q'(s',a')
        next_Q = self.target_model(next_states).squeeze().max(1)[0]  # [64]
        expected_Q = rewards + self.discount_factor * next_Q * (1 - dones)  # [64]
        expected_Q = expected_Q.to(self.device)

        # Loss
        # PER weight
        abs_td_error = F.smooth_l1_loss(curr_Q, expected_Q.detach(), reduction="none")
        loss = torch.mean(abs_td_error * weights)

        # PER: update priorities:
        new_priorities = abs_td_error.detach().cpu().numpy() + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

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
    EPOCHS = 2000
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
            "replay_period": agent.replay_period, # add replay period
            "per_beta": agent.memory.beta,
            "epochs": EPOCHS,
            "target_score": TARGET_SCORE
        },
        notes="Add plot score vs. epoch"
    )
    # individual x axis
    wandb.define_metric("epoch_step")
    wandb.define_metric("score_epoch", step_metric='epoch_step')

    for epoch in range(EPOCHS):
        done = False
        turncated = False
        score = 0

        state, _ = env.reset()
        while not(done or turncated):
            action = agent.get_action(state)
            next_state, reward, done, turncated, info = env.step(action)
            # agent.memory.append((state, action, reward, next_state, done))
            agent.memory.store(state, action, reward, next_state, done)
            if len(agent.memory) >= agent.train_start:
                agent.train()
            score += reward
            state = next_state

            if agent.step_counter % 100 == 0:
                wandb.log({"score": score, "epsilon": agent.epsilon, "step_counter": agent.step_counter, "per_beta": agent.memory.beta})

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
                    f'step_counter:{agent.step_counter}, '
                    f'per_beta:{agent.memory.beta}'
                )
                wandb.log({"avg_score": avg_score, "epoch_step": epoch, "score_epoch": score})
    
        if avg_score > TARGET_SCORE:
            print(f"Solved in episode: {epoch + 1}")
            break
    
    def plot(scores, episodes):
        import matplotlib.pyplot as plt
        plt.plot(episodes, scores)
        plt.title('Dueling DQN with PER For LunarLander-v3')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid()
        plt.show()
        # plt.savefig('Score vs. Episode.png')

    plot(scores, episodes)
    mean_score, std_score = agent.evaluate(num_episodes=5, render=True)
    print(f"Evaluated Result(Mean Score: {mean_score:.3f}, Std Score: {std_score:.3f})")
    wandb.log({"mean_score": mean_score, "std_score": std_score})
    
if __name__ == "__main__":
    main()
        
        
    