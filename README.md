# lunar-lander

Simplest implementation of lunar-lander

## Implementations

12/7(SAT)/2024

| Implementation                   | Status | Solved Episode | Assigned To |
|----------------------------------|--------|----------------|-------------|
| original DQN                     | [x]    | 717            | 배영민        |
| original DQN w/ Boltzman(tau=0.5)| [x]    | 306            | 배영민        |
| double DQN                       | [x]    | 327            | 배영민        |
| dueling DQN                      | [x]    | 408            | 임동휘        |
| double dueling DQN               | [x]    | 328            | 임동휘        |
| original DQN with PER            | [x]    | 471            | 김범수        |
| double DQN with PER              | [x]    | 453            | 김범수        |
| dueling DQN with PER             | [x]    | 671            | 이재영        |
| double dueling DQN with PER      | [x]    | 455            | 이재영        |
| A2C(n-step bootstrapping version)| [x]    | 6587           | 이용환        |

12/14(SAT)/2024

| Implementation                  | Status | Solved Episode | Assigned To |
|---------------------------------|--------|----------------|-------------|
| A2C(Monte Carlo version)        | [ ]    | N/A            | 윤효경        |
| A2C(TD(0) version)              | [ ]    | N/A            | 윤효경        |
| A2C(GAE version)                | [ ]    | N/A            | 윤효경        |
| DDPG                            | [ ]    | N/A            | 한일영        |
| PPO (w/o entropy, gae)          | [ ]    | N/A            | 지민기        |

12/21(SAT)/2024

| Implementation                  | Status | Solved Episode | Assigned To |
|---------------------------------|--------|----------------|-------------|
| A3C                             | [ ]    | N/A            | 이동훈      |
| Distributed DQN                 | [ ]    | N/A            | 이승수      |
| Distributed PPO                 | [ ]    | N/A            | 이경민      |

## How to make benchmarks

- `lunar-lander-v3` 기준으로 연속 30개 에피소드 평균 스코어가 260을 넘기면 solved로 간주합니다.

## Code convention

코드 작성시 유의사항
RL 알고리즘 구현하실 때, 모두가 쉽게 공유하기 위해 다음을 지켜주시기 바랍니다.

1. 하나의 파이썬 파일(`.py`)로 작성한다.
2. "최대한 단순하게 "작성한다.
3. 가급적 자세한 각주를 붙인다.

아시다시피 비교적 간단한 작업을 여러 디렉토리에 분산된 `.py` 파일들로 만들면 뜯어보는데 너무 시간이 소모됩니다. 

## Installation

```bash
conda install gymnasium-box2d wandb -c conda-forge -y
```

### Optional-wandb

```
wandb login
```

## Reference

- [keras-rl](https://github.com/keras-rl/keras-rl)
- [clean-rl](https://github.com/vwxyzjn/cleanrl)
- [udacity-deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning)
- [yandex-dataschool-practical-rL](https://github.com/yandexdataschool/Practical_RL)
- [curt-park-rainbow-rl](https://github.com/Curt-Park/rainbow-is-all-you-need)
- [AI4Finance-Foundation-elegant-rl](https://github.com/AI4Finance-Foundation/ElegantRL)