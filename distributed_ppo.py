import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import random
from collections import deque
import matplotlib.pyplot as plt
import copy
import os

###################################
# 1. ActorCritic 정의 (기존과 동일)
###################################
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

###################################
# 2. Worker 프로세스 정의
###################################
def worker_process(worker_id, 
                   child_conn, 
                   env_name, 
                   gamma, 
                   lam, 
                   rollout_length, 
                   hidden_dim, 
                   device):
    """
    Worker 프로세스:
      - 메인 프로세스(learner)에서 모델 파라미터를 수신
      - 환경에서 샘플(trajectory) 수집 후, 샘플을 메인 프로세스로 전송
      - 종료 신호가 오면 종료
    """
    
    # 각 워커가 자체적으로 환경 하나씩 가짐
    env = gym.make(env_name)
    
    # CPU 사용을 가정 (원한다면 device='cuda'로 설정 가능)
    # Worker에서 모델을 초기화하지만, 실제로는 파라미터를 메인에서만 업데이트하고
    # Worker들은 매 iteration에 메인에서 받은 파라미터를 load_state_dict로 갱신해서 사용
    model = ActorCritic(env.observation_space.shape[0],
                        env.action_space.n,
                        hidden_dim).to(device)

    def select_action(state):
        """
        Worker에서 액션을 샘플링하기 위한 함수.
        """
        state_t = torch.FloatTensor(state).to(device)
        policy, value = model(state_t)
        policy = nn.Softmax(dim=-1)(policy)
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    
    def compute_advantages(rewards, values, dones):
        """
        GAE 계산. (간단히 그대로 가져옴)
        """
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    while True:
        # 메인 프로세스(learner)에서 메시지 받음
        message = child_conn.recv()

        if message[0] == 'UPDATE_PARAMS':
            # 파라미터 업데이트
            new_params = message[1]
            model.load_state_dict(new_params)

        elif message[0] == 'COLLECT_ROLLOUT':
            # roll-out 수집 시작
            states = []
            actions = []
            log_probs = []
            values = []
            rewards = []
            dones = []
            state = env.reset()
            done = False

            while not done and len(states) < rollout_length:
                action, log_prob, value = select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(done)

                state = next_state

            # 마지막 상태의 value (bootstrap) 계산
            if not done:
                _, _, next_value = select_action(state)
            else:
                next_value = 0.0
            values.append(next_value)

            advantages = compute_advantages(rewards, values, dones)
            returns = [advantages[i] + values[i] for i in range(len(rewards))]

            # 수집한 rollout 데이터를 메인 프로세스로 보냄
            rollout_data = {
                'states': states,
                'actions': actions,
                'log_probs': log_probs,
                'values': values[:-1],  # 마지막 value는 bootstrap용이므로 제외
                'rewards': rewards,
                'dones': dones,
                'returns': returns,
            }
            child_conn.send(rollout_data)

        elif message[0] == 'TERMINATE':
            # 종료
            env.close()
            child_conn.close()
            break

###################################
# 3. Learner (메인) PPO 클래스 정의
###################################
class DistributedPPO:
    def __init__(
        self,
        env_name='LunarLander-v2',
        num_workers=4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        entropy_coefficient=0.01,
        lr=3e-4,
        epochs=10,
        batch_size=64,
        rollout_length=1024,
        hidden_dim=256,
        max_episodes=1000,
        device='cpu'
    ):
        self.env_name = env_name
        self.num_workers = num_workers
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coefficient = entropy_coefficient
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.hidden_dim = hidden_dim
        self.max_episodes = max_episodes
        self.device = device

        # learner(메인)가 모델 하나만 유지, 업데이트
        dummy_env = gym.make(env_name)
        self.model = ActorCritic(dummy_env.observation_space.shape[0],
                                 dummy_env.action_space.n,
                                 hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = []
        dummy_env.close()

        # 멀티프로세싱 관련 초기화
        self.workers = []
        self.parent_conns = []
        self.child_conns = []

        # Worker 프로세스 생성
        for i in range(num_workers):
            parent_conn, child_conn = mp.Pipe()
            w = mp.Process(target=worker_process,
                           args=(i,
                                 child_conn,
                                 self.env_name,
                                 self.gamma,
                                 self.lam,
                                 self.rollout_length,
                                 self.hidden_dim,
                                 self.device))
            w.start()
            self.workers.append(w)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

    def broadcast_parameters(self):
        """
        Learner의 최신 모델 파라미터를 모든 워커에게 전송
        """
        params = self.model.state_dict()
        for parent_conn in self.parent_conns:
            parent_conn.send(('UPDATE_PARAMS', params))

    def collect_rollouts(self):
        """
        모든 워커에게 rollout 수집 명령을 보내고, 그 결과를 수집
        """
        # 각 워커에게 rollout 수집 요청
        for parent_conn in self.parent_conns:
            parent_conn.send(('COLLECT_ROLLOUT',))

        # 워커들의 rollout 결과를 수신
        all_rollouts = []
        for parent_conn in self.parent_conns:
            rollout_data = parent_conn.recv()
            all_rollouts.append(rollout_data)

        return all_rollouts

    def make_batch(self, all_rollouts):
        """
        여러 Worker에서 수집된 rollout들을 하나로 합쳐서 학습에 쓸 수 있도록 정리
        """
        states = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []

        for rollout in all_rollouts:
            s = rollout['states']
            a = rollout['actions']
            lp = rollout['log_probs']
            r = rollout['returns']
            v = rollout['values']

            # advantage는 r - v 로도 쓸 수 있으나, 위에서 GAE를 이미 계산했으므로
            # rollout['advantages']가 있다고 가정하면 그대로 사용할 수도 있음.
            # 여기서는 returns와 values를 이용해서 advantage를 구하겠습니다.
            adv = []
            for ret, val in zip(r, v):
                adv.append(ret - val)

            states.extend(s)
            actions.extend(a)
            old_log_probs.extend(lp)
            returns.extend(r)
            advantages.extend(adv)

        # 파이토치 텐서로 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        return states, actions, old_log_probs, returns, advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        PPO 업데이트 (기존 코드와 유사)
        """
        for _ in range(self.epochs):
            # 미니배치 단위로 업데이트
            for i in range(0, len(states), self.batch_size):
                end = i + self.batch_size
                batch_obs = states[i:end]
                batch_actions = actions[i:end]
                batch_old_log_probs = old_log_probs[i:end]
                batch_returns = returns[i:end]
                batch_advantages = advantages[i:end]

                policy, value = self.model(batch_obs)
                policy = nn.Softmax(dim=-1)(policy)
                dist = torch.distributions.Categorical(policy)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio,
                                    1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coefficient * entropy

                critic_loss = (batch_returns - value.squeeze()).pow(2).mean()

                loss = actor_loss + critic_loss * 0.5

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self):
        """
        메인 학습 루프
        """
        episode_rewards = []
        score_history = deque(maxlen=100)
        best_score = -999999
        total_episodes = 0

        # 초기 파라미터 브로드캐스트
        self.broadcast_parameters()

        while total_episodes < self.max_episodes:
            # 1) 워커로부터 rollouts 수집
            all_rollouts = self.collect_rollouts()
            
            # 실제로는 각 rollout마다 몇 개 에피소드가 끝났는지 추적해야 함.
            # 예시 상에서는 rollout_length 동안 진행했으니 그 안에서 일부 에피소드가 끝날 수 있음.
            # 여기서는 간단히 episode_reward의 평균만 따로 로깅
            # (일반 PPO처럼 한 에피소드씩 돌리는 게 아니라 여러 에피소드가 섞임)

            # rollouts에서 reward 합계를 구해보자(간단 로그용)
            for rollout in all_rollouts:
                # 한 roll-out에 여러 에피소드가 섞일 수 있지만,
                # 여기서는 done == True 만나면 그 에피소드의 reward만 부분 합산
                sum_reward = 0
                ep_count = 0
                running_reward = 0
                for r, d in zip(rollout['rewards'], rollout['dones']):
                    running_reward += r
                    if d:
                        sum_reward += running_reward
                        running_reward = 0
                        ep_count += 1
                if ep_count > 0:
                    avg_ep_reward = sum_reward / ep_count
                    score_history.append(avg_ep_reward)
                    episode_rewards.append(avg_ep_reward)
                    total_episodes += ep_count

            # 2) PPO 업데이트용 배치 생성
            states, actions, old_log_probs, returns, advantages = self.make_batch(all_rollouts)
            # 3) 학습(업데이트)
            self.update(states, actions, old_log_probs, returns, advantages)
            # 4) 업데이트된 파라미터를 브로드캐스트
            self.broadcast_parameters()

            if len(score_history) > 0:
                mean_score = np.mean(score_history)
            else:
                mean_score = 0.0

            if mean_score > best_score:
                best_score = mean_score
                torch.save(self.model.state_dict(), 'best_distributed_model.pth')

            print(f"[Learner] Total Episodes: {total_episodes}, "
                  f"Mean Score (last 100): {mean_score:.2f}")

        # 모든 워커 종료
        for parent_conn in self.parent_conns:
            parent_conn.send(('TERMINATE',))
        
        for w in self.workers:
            w.join()

        return episode_rewards

###################################
# 4. main 함수
###################################
def main():
    mp.set_start_method('spawn')  # 또는 'forkserver' (OS에 따라 설정 필요)

    # 분산 PPO 학습 인스턴스 생성
    learner = DistributedPPO(
        env_name='LunarLander-v2',
        num_workers=4,         # 병렬 환경 개수
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        entropy_coefficient=0.01,
        lr=3e-4,
        epochs=10,
        batch_size=64,
        rollout_length=1024,
        hidden_dim=256,
        max_episodes=2000,
        device='cpu'  # 가능하다면 'cuda' 사용
    )

    rewards = learner.train()
    plt.plot(rewards)
    plt.xlabel("Episodes (approx, aggregated)")
    plt.ylabel("Rewards")
    plt.title("Distributed PPO - LunarLander")
    plt.savefig("distributed_ppo_lunarlander.png")
    plt.show()

if __name__ == "__main__":
    main()
