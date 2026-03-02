"""
Utility functions for plotting training performance
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.show()
