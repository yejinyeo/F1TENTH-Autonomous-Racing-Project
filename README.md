# F1TENTH-Autonomous-Racing-Project
[📄 Project presentation PDF](https://github.com/yejinyeo/F1TENTH-Autonomous-Racing-Project/blob/main/%5B5-2%5D%20RL_F1TENTH_project_presentation.pdf)

## Introduction
본 프로젝트는 **DQN 기반 강화학습 알고리즘을 활용해 F1TENTH 자율주행 플랫폼에서 lap time 최소화를 목표로 한 연구 프로젝트**입니다. Multi-step, Double, Dueling DQN 등 다양한 variants 및 Reward Shaping, PER을 ablation study 실험을 통해 성능을 비교 분석하였고, 실제 주행 영상을 통해 모델 성능을 검증하였습니다.

## Key Features
- **DQN Variants 실험**: Multi-step, Double, Dueling, Dueling + Double DQN
- **Reward Shaping / PER 적용 여부별 ablation study 진행**
- **Train/Eval Speed Parameter 미세 조정 전략으로 lap time 단축**
- **Oschersleben 및 Easy 맵 실주행 lap time 검증**

## Implementation

### 1️⃣ Baseline 및 알고리즘
- DQN, Multi-step DQN, Double DQN, Dueling DQN, Dueling + Double DQN
- Prioritized Experience Replay (PER) 적용 및 annealing 스케줄 사용

### 2️⃣ 강화 기법
- Reward Shaping
- PER
- Reward Shaping + PER 조합
- PER의 β 계수 0.4 → 1.0 점진적 증가로 탐험 → 수렴 유도

### 3️⃣ Ablation Study
- 모델 구조별 성능 비교
- 강화 기법별 성능 비교
- Train 환경과 Eval 환경 속도 미세 조정으로 lap time 단축



