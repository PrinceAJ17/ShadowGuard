"""
ShadowGuard Security Robot - Fast Accurate SLAM with Sharp Mapping
Now with Reinforcement Learning (Q-learning) for optimal patrol strategies
"""

from controller import Robot, Camera, Lidar, GPS, Compass, Display, Emitter
import numpy as np
import math
import random
import cv2
import threading
import time
from collections import deque
import json
import os

class QLearningAgent:
    """Reinforcement Learning agent using Q-learning for patrol optimization"""
    
    def __init__(self, state_space_size, action_space_size):
        self.state_space = state_space_size
        self.action_space = action_space_size
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
        # Initialize Q-table
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # Learning statistics
        self.total_rewards = 0
        self.episode_rewards = 0
        self.episode_count = 0
        self.steps = 0
        self.learning_history = []
        
        # Action meanings
        self.action_meanings = [
            "Explore Forward",
            "Turn Left",
            "Turn Right", 
            "Move to Center",
            "Check Corners",
            "Slow Patrol"
        ]
        
        # Load previous Q-table if exists
        self.load_q_table()
        
        print(f"RL Agent initialized: {state_space_size} states, {action_space_size} actions")
    
    def discretize_state(self, robot_x, robot_y, robot_theta, intruder_detected, obstacle_near):
        """Convert continuous state to discrete state index"""
        # Discretize position into grid cells (4x4 grid)
        grid_size = 4
        x_bin = int(np.clip((robot_x + 6) / 12 * grid_size, 0, grid_size-1))
        y_bin = int(np.clip((robot_y + 5) / 10 * grid_size, 0, grid_size-1))
        
        # Discretize orientation (4 directions)
        theta_bin = int((robot_theta + math.pi) / (2 * math.pi) * 4) % 4
        
        # Combine all features
        state = (
            x_bin * grid_size * 4 * 2 * 2 +
            y_bin * 4 * 2 * 2 +
            theta_bin * 2 * 2 +
            int(intruder_detected) * 2 +
            int(obstacle_near)
        )
        
        return min(state, self.state_space - 1)
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_space - 1)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state, action] = new_q
        
        # Update statistics
        self.episode_rewards += reward
        self.steps += 1
        
        return new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def end_episode(self):
        """End current episode and record statistics"""
        self.total_rewards += self.episode_rewards
        self.episode_count += 1
        
        self.learning_history.append({
            'episode': self.episode_count,
            'reward': self.episode_rewards,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'avg_q': np.mean(self.q_table)
        })
        
        # Reset episode stats
        self.episode_rewards = 0
        self.steps = 0
        
        # Save Q-table periodically
        if self.episode_count % 10 == 0:
            self.save_q_table()
        
        # Decay exploration rate
        self.decay_epsilon()
    
    def save_q_table(self):
        """Save Q-table to file"""
        try:
            data = {
                'q_table': self.q_table.tolist(),
                'episode_count': self.episode_count,
                'total_rewards': self.total_rewards,
                'epsilon': self.epsilon
            }
            with open('q_table.json', 'w') as f:
                json.dump(data, f)
            print(f"Q-table saved (Episode {self.episode_count})")
        except Exception as e:
            print(f"Error saving Q-table: {e}")
    
    def load_q_table(self):
        """Load Q-table from file"""
        try:
            if os.path.exists('q_table.json'):
                with open('q_table.json', 'r') as f:
                    data = json.load(f)
                self.q_table = np.array(data['q_table'])
                self.episode_count = data.get('episode_count', 0)
                self.total_rewards = data.get('total_rewards', 0)
                self.epsilon = data.get('epsilon', 0.3)
                print(f"Q-table loaded (Episode {self.episode_count})")
        except Exception as e:
            print(f"Error loading Q-table: {e}")