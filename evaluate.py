"""
Evaluation Script
Tests trained PPO agent and computes win rate
"""

import gymnasium as gym
import chefshatgym
from stable_baselines3 import PPO
import numpy as np

# Load environment
env = gym.make("ChefHat-v0")

# Load trained model
model = PPO.load("models/ppo_chefshat")

episodes = 100
wins = 0

print("Evaluating trained agent...")

for episode in range(episodes):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    # Assuming positive reward = win
    if reward > 0:
        wins += 1

win_rate = wins / episodes

print(f"Win Rate over {episodes} episodes: {win_rate}")
