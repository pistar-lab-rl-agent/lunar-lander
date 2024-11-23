# lunar-lander

Simplest implementation of lunar-lander

## Implementations

### 12/7(SAT)/2024

| Implementation                   | Status | Solved Episode |
|----------------------------------|--------|----------------|
| original DQN                     | [x]    | 688            |
| double DQN                       | [x]    | N/A            |
| dueling DQN                      | [ ]    | N/A            |
| double dueling DQN               | [ ]    | N/A            |
| original DQN with PER            | [ ]    | N/A            |
| double DQN with PER              | [ ]    | N/A            |
| dueling DQN with PER             | [ ]    | N/A            |
| double dueling DQN with PER      | [ ]    | N/A            |
| A2C(n-step bootstrapping version)| [ ]    | N/A            |

### 12/14(SAT)/2024

| Implementation                  | Status | Solved Episode |
|---------------------------------|--------|----------------|
| A2C(Monte Carlo version)        | [ ]    | N/A            |
| A2C(TD(0) version)              | [ ]    | N/A            |
| A2C(GAE version)                | [ ]    | N/A            |
| DDPG                            | [ ]    | N/A            |
| PPO                             | [ ]    | N/A            |

### 12/21(SAT)/2024

| Implementation                  | Status | Solved Episode |
|---------------------------------|--------|----------------|
| A3C                             | [ ]    | N/A            |
| Distributed DQN                 | [ ]    | N/A            |
| Distributed PPO                 | [ ]    | N/A            |

## How to make benchmarks

- `lunar-lander-v3` 기준으로 연속 30개 에피소드 평균 스코어가 200을 넘기면 solved로 간주합니다.