"""
Task 2 - Reinforcement Learning
Chef's Hat Gym Environment
Algorithm: PPO (Proximal Policy Optimization)

This script:
1. Creates Chef's Hat environment
2. Trains PPO agent
3. Saves trained model
4. Logs learning progress
"""

import gymnasium as gym
import chefshatgym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

# ---------------------------------------------
# Create Environment
# ---------------------------------------------

env = gym.make("ChefHat-v0")

# Wrap environment with Monitor for logging
env = Monitor(env)

# ---------------------------------------------
# Create PPO Model
# ---------------------------------------------

model = PPO(
    policy="MlpPolicy",          # Multi-layer perceptron policy
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    tensorboard_log="./ppo_logs/"
)

# ---------------------------------------------
# Evaluation Callback
# Automatically evaluates during training
# ---------------------------------------------

eval_env = gym.make("ChefHat-v0")
eval_env = Monitor(eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# ---------------------------------------------
# Train Agent
# ---------------------------------------------

print("Starting PPO training...")

model.learn(
    total_timesteps=100000,   # Adjust for experimentation
    callback=eval_callback
)

# ---------------------------------------------
# Save Final Model
# ---------------------------------------------

os.makedirs("models", exist_ok=True)
model.save("models/ppo_chefshat")

print("Training complete. Model saved.")
