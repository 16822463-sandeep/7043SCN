# Reinforcement Learning Agent – Chef’s Hat Gym (PPO)

## Module
7043SCN – Generative AI and Reinforcement Learning  
Task 2 – Reinforcement Learning

---

## Assigned Variant

- **Student ID:** 16822463  
- **Student ID mod 7:** 16822463 % 7 = 0  
- **Variant:** Opponent Modelling Variant  

This project investigates the impact of different opponent behaviours on PPO agent performance in a competitive multi-agent environment.

---

## Environment

- Official Chef’s Hat Gym environment  
- Multi-agent competitive card game  
- Discrete and variable action space  
- Sparse and delayed rewards  
- Stochastic opponent behaviour  
- Non-stationary learning dynamics  

GitHub: https://github.com/pablovin/ChefsHatGYM  
Docs: https://chefshatgym.readthedocs.io/en/latest/

---

## RL Algorithm

- Algorithm: Proximal Policy Optimization (PPO)
- Library: Stable-Baselines3
- Policy: MlpPolicy

### Justification
- Stable training via clipped objective
- Works well in discrete action spaces
- Suitable for stochastic multi-agent settings
- Good exploration–exploitation balance

---

## State Representation

- Observation vector provided directly by environment
- No manual feature engineering
- Invalid actions masked by environment
- MLP used for policy and value approximation

---

## Reward Usage

- Sparse reward structure
- Positive reward = win
- Neutral/negative reward = loss
- No reward shaping applied

---

## Training Setup

- Learning rate: 3e-4
- Batch size: 64
- Gamma (discount factor): 0.99
- Rollout steps: 2048
- Total timesteps: 100,000
- Evaluation episodes: 100
- Monitor wrapper for logging
- EvalCallback for periodic evaluation

---

## Opponent Modelling Experiments

To satisfy the Opponent Modelling Variant:

### Experiment 1 – PPO vs Random Opponent
- Trained and evaluated against default stochastic opponent
- Measured win rate over 100 episodes

### Experiment 2 – PPO vs Alternative Opponent Configuration
- Evaluated under different opponent behaviour (if configured)
- Compared win rate variation

### Experiment 3 – Random Seed Analysis
- Training repeated with different seeds
- Analysed performance stability
- Observed non-stationarity effects

---

## Evaluation Metrics

- Win rate
- Cumulative reward
- Learning curve stability
- Performance variance across seeds

---

## Key Findings

- Performance improves over training
- Strong against random opponents
- Sensitive to opponent behaviour changes
- Multi-agent non-stationarity affects convergence

---

## Limitations

- No explicit opponent prediction mechanism
- No recurrent/memory-based architecture
- Sparse reward complicates credit assignment
- Seed sensitivity impacts stability

---
