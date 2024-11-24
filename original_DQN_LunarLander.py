# DQN(original version) For LunarLander-v3
# It takes about 2 minutes to solve the environment on M2-max
# Implemented by YH Lee

import random
import numpy as np
import time
import gymnasium as gym
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for random, torch modules
random.seed(17)
torch.manual_seed(17)

class DQNAgent:
    def __init__(self, state_size, action_size, load_model=False):
        self.load_model = load_model
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
        self.target_update_period = 200

        # Replay Buffer
        self.memory = deque(maxlen=10000)

        # Model and Target Model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initial Synchronization of Target Model
        self.update_target_model()

        if self.load_model:
            self.model.load_state_dict(torch.load("./save_model/lunarlander_v3_dqn_trained_01.pth"))

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.FloatTensor(state)
            with torch.no_grad():
                q_value = self.model(state)
            return q_value.argmax().item()

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([x[0] for x in mini_batch]))  # [64,8]
        actions = torch.LongTensor(np.array([x[1] for x in mini_batch]))  # [64]
        rewards = torch.FloatTensor(np.array([x[2] for x in mini_batch]))  # [64]
        next_states = torch.FloatTensor(np.array([x[3] for x in mini_batch]))  # [64,8]
        dones = torch.FloatTensor(np.array([x[4] for x in mini_batch]))  # [64]

        # Current Q Values: Q(s,a)
        curr_Q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # [64]

        # Next Q Values: max_a' Q'(s',a')
        next_Q = self.target_model(next_states).squeeze().max(1)[0]  # [64]
        expected_Q = rewards + self.discount_factor * next_Q * (1 - dones)  # [64]

        # Loss
        loss = nn.MSELoss()(curr_Q, expected_Q.detach())

        # Update Parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot_scores(self, episodes, scores):
        plt.plot(episodes, scores)
        plt.title('original DQN For LunarLander-v3')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid()
        plt.show()

    def evaluate(self, num_episodes=5, render=False):
        env = gym.make('LunarLander-v3', render_mode='human' if render else None)
        self.model.eval()
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


if __name__ == "__main__":

    t1 = time.time()

    env = gym.make('LunarLander-v3', render_mode="rgb_array")

    # Set random seed for env
    env.reset(seed=7)
    env.action_space.seed(7)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Instance of DQN agent
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    EPOCHS = 2000
    TARGET_SCORE = 260

    # Train Loop
    for e in range(EPOCHS):
        done = False
        truncated = False
        score = 0
        step_counter = 0  # max=1000 -> done=True

        # Initialize env
        state, _ = env.reset()

        while not (done or truncated):
            # determine action in state
            action = agent.get_action(state)

            # transition to next step by env
            next_state, reward, done, truncated, info = env.step(action)
            step_counter += 1

            # save (s, a, r, s', d) in replay buffer
            agent.append_sample(state, action, reward, next_state, done)

            # train at every step
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            # Synchronize target model with model
            if step_counter % agent.target_update_period == 0:
                agent.update_target_model()

            if done or truncated:
                # # Synchronize target model with model at the end of an episode
                agent.update_target_model()

                # Record score and print out results
                scores.append(score)
                episodes.append(e)
                avg_score = np.mean(scores[-min(30, len(scores)):])
                print(f'episode:{e}, score:{score:.3f}, avg_score:{avg_score:.3f}, epsilon:{agent.epsilon:.3f}, step_counter:{step_counter}')

        if avg_score > TARGET_SCORE:
            #         torch.save(agent.model.state_dict(), "./save_model/lunarlander_v3_dqn_trained_01.pth")
            #                 np.savez_compressed('perf_rec_01.npz', np.array(scores))
            t2 = time.time()
            print("The env is solved!")
            print("Elapsed Time(min):", round((t2 - t1) / 60, 2))
            break

    # Print scores vs episodes
    agent.plot_scores(episodes,scores)

    # Evaluate trained model
    mean_score, std_score = agent.evaluate(num_episodes=5, render=True)
    print(f'Evaluation - Mean Reward: {mean_score:.2f} ± {std_score:.2f}')



