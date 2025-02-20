import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import random
from collections import deque
import matplotlib.pyplot as plt

###################################
# 1. DQN 네트워크 정의
###################################
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

###################################
# 2. Worker 프로세스 정의
###################################
def worker_process(
    worker_id, 
    child_conn, 
    env_name, 
    initial_epsilon, 
    epsilon_decay,  # Parameter kept for compatibility.
    epsilon_min,    # Parameter kept for compatibility.
    rollout_length, 
    hidden_dim, 
    device
):
    """
    Worker 프로세스:
      - 메인 프로세스에서 모델 파라미터와 중앙 epsilon 값을 수신
      - 중앙 epsilon 값에 따라 ε-greedy 정책으로 환경에서 transition을 수집하고 메인 프로세스로 전송
      - 종료 신호가 오면 종료
    """
    env = gym.make(env_name)
    model = DQN(env.observation_space.shape[0],
                env.action_space.n,
                hidden_dim).to(device)
    epsilon = initial_epsilon  # 초기 epsilon

    def select_action(state, epsilon):
        state_t = torch.FloatTensor(state).to(device)
        if random.random() < epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            q_values = model(state_t)
        return q_values.argmax().item()

    while True:
        message = child_conn.recv()

        if message[0] == 'UPDATE_PARAMS':
            new_params = message[1]
            model.load_state_dict(new_params)
            # 중앙에서 전송받은 epsilon 값으로 업데이트.
            if len(message) > 2:
                epsilon = message[2]

        elif message[0] == 'COLLECT_ROLLOUT':
            transitions = []  # 각 transition: (state, action, reward, next_state, terminal)
            episode_scores = []  # 완결된 episode의 점수를 저장
            episode_score = 0.0

            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            for _ in range(rollout_length):
                action = select_action(state, epsilon)
                next_step = env.step(action)
                # gym 버전에 따라 반환 값 처리
                if len(next_step) == 5:
                    next_state, reward, done, truncated, info = next_step
                else:
                    next_state, reward, done, info = next_step
                    truncated = False
                terminal = done or truncated
                transitions.append((state, action, reward, next_state, float(terminal)))
                episode_score += reward
                state = next_state
                if terminal:
                    episode_scores.append(episode_score)
                    episode_score = 0.0
                    state = env.reset()
                    if isinstance(state, tuple):
                        state = state[0]
            child_conn.send((transitions, episode_scores))

        elif message[0] == 'TERMINATE':
            env.close()
            child_conn.close()
            break

###################################
# 3. Learner (메인) Distributed DQN 클래스 정의
###################################
class DistributedDQN:
    def __init__(
        self,
        env_name='LunarLander-v2',
        num_workers=4,
        epsilon=1.0,
        epsilon_decay=0.99995,
        epsilon_min=0.01,
        rollout_length=1024,
        hidden_dim=256,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        replay_buffer_size=100000,
        target_update_interval=1000,
        max_episodes=1000,
        target_score=200,
        device='cpu'
    ):
        self.env_name = env_name
        self.num_workers = num_workers
        self.epsilon = epsilon  # 중앙 epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rollout_length = rollout_length
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.target_update_interval = target_update_interval
        self.max_episodes = max_episodes
        self.target_score = target_score
        self.device = device

        # 환경 정보 획득
        dummy_env = gym.make(env_name)
        self.state_dim = dummy_env.observation_space.shape[0]
        self.action_dim = dummy_env.action_space.n
        dummy_env.close()

        # 온라인 Q-네트워크와 타겟 네트워크 초기화
        self.policy_net = DQN(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_net = DQN(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay buffer (deque 사용)
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # 멀티프로세싱 관련 초기화: 워커 프로세스 생성
        self.workers = []
        self.parent_conns = []
        self.child_conns = []

        for i in range(num_workers):
            parent_conn, child_conn = mp.Pipe()
            w = mp.Process(target=worker_process,
                           args=(i,
                                 child_conn,
                                 self.env_name,
                                 self.epsilon,
                                 self.epsilon_decay,
                                 self.epsilon_min,
                                 self.rollout_length,
                                 self.hidden_dim,
                                 self.device))
            w.start()
            self.workers.append(w)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

        self.total_steps = 0

    def broadcast_parameters(self):
        """
        Learner의 최신 모델 파라미터와 중앙 epsilon 값을 모든 워커에게 전송
        """
        params = self.policy_net.state_dict()
        for parent_conn in self.parent_conns:
            parent_conn.send(('UPDATE_PARAMS', params, self.epsilon))

    def collect_rollouts(self):
        """
        모든 워커에게 rollout 수집 명령을 보내고 전송받은 transition과 episode 점수를 집계함
        """
        for parent_conn in self.parent_conns:
            parent_conn.send(('COLLECT_ROLLOUT',))
        all_transitions = []
        all_episode_scores = []
        for parent_conn in self.parent_conns:
            transitions, episode_scores = parent_conn.recv()
            all_transitions.extend(transitions)
            all_episode_scores.extend(episode_scores)
        return all_transitions, all_episode_scores

    def update_replay_buffer(self, transitions):
        """
        수집한 transition들을 replay buffer에 추가
        """
        for trans in transitions:
            self.replay_buffer.append(trans)

    def sample_batch(self):
        """
        replay buffer에서 미니배치 샘플링
        """
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        return states, actions, rewards, next_states, dones

    def update(self):
        """
        DQN 업데이트: 미니배치에 대해 Q값 업데이트 수행
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.sample_batch()
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)
        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        """
        메인 학습 루프:
          - 모든 워커로부터 transition과 episode 점수 수집
          - Replay buffer 업데이트 및 미니배치로 Q-network 학습
          - 타겟 네트워크 업데이트 및 최근 30 episode의 평균 점수를 계산 및 로깅
          - 평균 점수가 목표치를 달성하면 조기 종료
        """
        losses = []
        all_scores = []  # 각 episode 점수 저장
        iteration = 0   # rollout 횟수

        # 최초 파라미터와 epsilon 전송
        self.broadcast_parameters()

        while iteration < self.max_episodes:
            transitions, episode_scores = self.collect_rollouts()
            self.update_replay_buffer(transitions)
            all_scores.extend(episode_scores)

            # 수집한 transition 개수에 따라 여러 번 업데이트 수행
            for _ in range(len(transitions) // self.batch_size):
                loss = self.update()
                if loss is not None:
                    losses.append(loss)
                self.total_steps += 1
                # 타겟 네트워크 주기적 업데이트
                if self.total_steps % self.target_update_interval == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # 최근 30 episode의 평균 점수 계산
            if all_scores:
                window = min(30, len(all_scores))
                avg_score = np.mean(all_scores[-window:])
            else:
                avg_score = 0.0

            print(f"[Learner] Iteration: {iteration}, Total Steps: {self.total_steps}, "
                  f"Avg Loss: {np.mean(losses[-100:]) if losses else 0.0:.4f}, "
                  f"Avg Score (last {window} episodes): {avg_score:.3f}, "
                  f"Replay Buffer Size: {len(self.replay_buffer)}")

            # 조기 종료: 목표 점수를 달성하였으면 종료
            if all_scores and avg_score >= self.target_score:
                print(f"Solved! Reached avg_score >= {self.target_score} at iteration {iteration}")
                break

            iteration += 1
            # 중앙 epsilon 서서히 감소
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.broadcast_parameters()

        # 모든 워커 프로세스 종료
        for parent_conn in self.parent_conns:
            parent_conn.send(('TERMINATE',))
        for w in self.workers:
            w.join()

        return losses, all_scores

###################################
# 4. main 함수
###################################
def main():
    mp.set_start_method('spawn')  # 또는 'forkserver' (OS에 따라 설정 필요)

    # Distributed DQN 학습 인스턴스 생성
    learner = DistributedDQN(
        env_name='LunarLander-v2',
        num_workers=8,         # 병렬 환경 개수
        epsilon=1.0,           # 중앙에서 관리하는 ε-greedy 탐색 확률
        epsilon_decay=0.99995,
        epsilon_min=0.01,
        rollout_length=1024,
        hidden_dim=128,
        gamma=0.99,
        lr=0.0005,
        batch_size=64,
        replay_buffer_size=200000,
        target_update_interval=500,
        max_episodes=2000,
        device='cpu'  # 'cuda' 사용 가능
    )

    losses, scores = learner.train()

    # 학습 과정에서의 손실(loss) 변화를 플롯
    plt.plot(losses)
    plt.xlabel("Update Iterations")
    plt.ylabel("Loss")
    plt.title("Distributed DQN Training Loss")
    plt.savefig("distributed_dqn_training_loss.png")
    plt.show()

if __name__ == "__main__":
    main()