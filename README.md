# lunar-lander

Simplest implementation of lunar-lander

## Implementations

12/7(SAT)/2024

| Implementation                   | Status | Solved Episode | Assigned To |
|----------------------------------|--------|----------------|-------------|
| original DQN                     | [x]    | 688            | 배영민        |
| double DQN                       | [x]    | 912            | 배영민        |
| dueling DQN                      | [ ]    | N/A            |             |
| double dueling DQN               | [ ]    | N/A            |             |
| original DQN with PER            | [ ]    | N/A            |             |
| double DQN with PER              | [ ]    | N/A            |             |
| dueling DQN with PER             | [ ]    | N/A            |             |
| double dueling DQN with PER      | [ ]    | N/A            |             |
| A2C(n-step bootstrapping version)| [ ]    | N/A            |             |

12/14(SAT)/2024

| Implementation                  | Status | Solved Episode | Assigned To |
|---------------------------------|--------|----------------|-------------|
| A2C(Monte Carlo version)        | [ ]    | N/A            |             |
| A2C(TD(0) version)              | [ ]    | N/A            |             |
| A2C(GAE version)                | [ ]    | N/A            |             |
| DDPG                            | [ ]    | N/A            |             |
| PPO                             | [ ]    | N/A            |             |

12/21(SAT)/2024

| Implementation                  | Status | Solved Episode | Assigned To |
|---------------------------------|--------|----------------|-------------|
| A3C                             | [ ]    | N/A            |             |
| Distributed DQN                 | [ ]    | N/A            |             |
| Distributed PPO                 | [ ]    | N/A            |             |

## How to make benchmarks

- `lunar-lander-v3` 기준으로 연속 30개 에피소드 평균 스코어가 200을 넘기면 solved로 간주합니다.

## Code convention

코드 작성시 유의사항
RL 알고리즘 구현하실 때, 모두가 쉽게 공유하기 위해 다음을 지켜주시기 바랍니다.

1. 하나의 쥬피터 노트북 파일(`ipynb`)로 작성한다.
2. "최대한 단순하게 "작성한다.
3. 가급적 자세한 각주를 붙인다.

아시다시피 비교적 간다한 작업을 여러 디렉토리에 분산된 `.py` 파일들로 만들면 뜯어보는데 너무 시간이 소모됩니다. 
쥬피터 노트북이 도저히 불편하신 분은 하나의 `.py` 파일로 만드시면 됩니다.

## Installation
```bash
conda install gymnasium-box2d -c conda-forge -y
```
