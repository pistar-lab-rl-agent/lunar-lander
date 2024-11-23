import random
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # 은닉층
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)  # 출력층

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trader:
    def __init__(self, state_size, action_size):
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 평가 모드로 설정
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # NumPy를 사용한 리플레이 버퍼 초기화
        self.memory_capacity = 10000
        self.state_memory = np.zeros((self.memory_capacity, state_size), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_capacity, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_capacity, dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_capacity, state_size), dtype=np.float32)
        self.done_memory = np.zeros(self.memory_capacity, dtype=np.bool_)
        self.mem_cntr = 0  # 현재 메모리 인덱스
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_min = 0.005  # 탐험(exploration)을 위한 엡실론
        self.epsilon_decay = 0.99995
        self.action_size = action_size
        self.steps_done = 0  # 학습 스텝 수

        # 로그를 위한 변수
        self.loss_history = []

    def get_action(self, state):
        self.steps_done += 1
        # Ensure state is a NumPy array with the correct shape
        if isinstance(state, tuple):
            state = state[0]  # Extract the first element if state is a tuple
        state = np.array(state, dtype=np.float32)
        state = state.reshape(1, -1)  # Reshape to (1, state_size)
        
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            return action
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
                return action

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.memory_capacity  # 순환 버퍼
        # Ensure state and next_state are 1D arrays
        if isinstance(state, tuple):
            state = state[0]  # Extract the first element if state is a tuple 
        state = np.array(state, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done
        self.mem_cntr += 1

    def train(self):
        # 메모리 크기가 1000 이상일 때에만 학습 진행
        if self.mem_cntr < 1000:
            return  # 충분한 메모리가 쌓일 때까지 대기

        max_mem = min(self.mem_cntr, self.memory_capacity)
        batch_indices = np.random.choice(max_mem, self.batch_size, replace=False)
        
        # NumPy 배열에서 배치 데이터 가져오기
        state_batch = self.state_memory[batch_indices]
        action_batch = self.action_memory[batch_indices]
        reward_batch = self.reward_memory[batch_indices]
        next_state_batch = self.next_state_memory[batch_indices]
        done_batch = self.done_memory[batch_indices]

        # 타겟 Q 값 계산
        target_q_values = calculate_target_q_values(
            self.policy_net, self.target_net, next_state_batch, reward_batch, done_batch, self.gamma
        )

        # Torch 텐서로 변환
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1)

        # 현재 Q 값 계산
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # 손실 계산
        loss = F.mse_loss(q_values, target_q_values)

        # 손실 값 로그에 저장
        self.loss_history.append(loss.item())

        # 역전파 및 옵티마이저 스텝
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 일정 간격으로 타겟 네트워크 업데이트
        if self.steps_done % 50 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def calculate_target_q_values(policy_net, target_net, next_state_batch, reward_batch, done_batch, gamma):
    # NumPy 배열을 Torch 텐서로 변환
    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
    done_batch = torch.tensor(done_batch.astype(np.float32))

    # 다음 상태에서의 최대 Q 값 계산
    with torch.no_grad():
        # Use policy_net to select the best action for the next state
        next_actions = policy_net(next_state_batch).argmax(1)

        # Use target_net to Evaluate the Q value of the selected action
        next_q_values = target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze()

    # 타겟 Q 값 계산
    target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

    return target_q_values


def main():
    env = gymnasium.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Trader(state_size, action_size)

    num_episodes = 1000 
    recent_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0  # 에피소드별 총 보상
        done = False
        step = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            recent_rewards.append(reward)
            step += 1

        if episode >= 30 and (episode + 1) % 100 == 0:
            _recent_rewards = [recent_rewards[i] for i in range(-30, 0)]
            if np.mean(_recent_rewards) > 200:
                print(f"Solved in episode: {episode + 1}")
                break

        avg_loss = np.mean(agent.loss_history[-agent.steps_done:])
        print(f"Episode {episode + 1} ended - Total Reward: {total_reward}, Average Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")

if __name__ == "__main__":
    main()