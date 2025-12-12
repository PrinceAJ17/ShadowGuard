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

class ShadowGuardSLAM:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Initialize devices
        self.camera = self.robot.getDevice('rgb_camera')
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        
        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)
        
        self.left_motor = self.robot.getDevice('left_motor')
        self.right_motor = self.robot.getDevice('right_motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        
        # Displays
        self.camera_display = self.robot.getDevice('camera_display')
        
        # Emitter
        self.emitter = self.robot.getDevice('emitter')
        if self.emitter:
            self.emitter.setChannel(1)
        
        # Fast, accurate SLAM Map
        self.map_size = 800  # Higher resolution for sharp edges
        self.map_resolution = 0.02  # 2cm per pixel
        self.map_center = self.map_size // 2
        
        # Map layers with thread locks
        self.map_lock = threading.Lock()
        self.obstacle_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        self.wall_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        
        # RL Map Layer (for visualization)
        self.rl_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # Display map (color)
        self.display_map = np.ones((self.map_size, self.map_size, 3), dtype=np.uint8) * 220  # Light gray
        
        # Thread-safe robot data
        self.trajectory = []
        self.trajectory_lock = threading.Lock()
        self.robot_speed = 3.0
        
        # Thread-safe intruder data
        self.intruder_data_lock = threading.Lock()
        self.intruder_detected = False
        self.intruder_position = None
        self.intruder_trajectory = []
        
        # Patrol parameters
        self.patrol_state = 'exploring'
        self.obstacle_detected = False
        self.avoid_timer = 0
        self.wander_angle = 0
        self.last_turn_time = 0
        
        # Fast mapping settings
        self.obstacle_threshold = 0.5
        self.wall_distance = 1.8
        self.map_update_counter = 0
        
        # Reinforcement Learning Agent
        state_space_size = 4 * 4 * 4 * 2 * 2  # 4x4 grid * 4 directions * 2 intruder states * 2 obstacle states
        action_space_size = 6  # 6 different patrol strategies
        self.rl_agent = QLearningAgent(state_space_size, action_space_size)
        
        # RL state tracking
        self.current_rl_state = None
        self.current_rl_action = None
        self.last_rl_update_time = 0
        self.rl_action_duration = 0
        self.rl_episode_start_time = 0
        
        # RL Visualization data
        self.rl_q_values = []
        self.rl_rewards_history = []
        self.rl_visited_cells = set()
        
        # OpenCV windows
        self.map_window_name = "ShadowGuard - Accurate SLAM Map"
        self.rl_window_name = "RL Learning Dashboard"
        self.show_map = True
        self.show_rl = True
        self.map_image = None
        
        print("ShadowGuard Fast SLAM Controller with RL Initialized")
        print("Mapping: Fast and accurate with sharp edges")
        print("Robot Speed: Normal (3.0 m/s)")
        print(f"RL Agent: {state_space_size} states, {action_space_size} actions")
        
    def world_to_map(self, x, y):
        """Fast coordinate conversion"""
        map_x = int(self.map_center + x / self.map_resolution)
        map_y = int(self.map_center - y / self.map_resolution)
        return np.clip(map_x, 0, self.map_size-1), np.clip(map_y, 0, self.map_size-1)
    
    def get_robot_pose(self):
        """Get current robot position and orientation"""
        pos = self.gps.getValues()
        compass_vals = self.compass.getValues()
        angle = math.atan2(compass_vals[0], compass_vals[1])
        return pos[0], pos[1], angle
    
    def fast_mapping(self, robot_x, robot_y, robot_theta, ranges):
        """Fast and accurate mapping with line-based obstacle detection"""
        with self.map_lock:
            robot_mx, robot_my = self.world_to_map(robot_x, robot_y)
            
            # Track visited cells for RL
            cell_x = int((robot_x + 6) / 12 * 20)  # 20x20 grid for RL visualization
            cell_y = int((robot_y + 5) / 10 * 20)
            self.rl_visited_cells.add((cell_x, cell_y))
            
            # Process LiDAR points
            num_points = len(ranges)
            for i, r in enumerate(ranges):
                if r < self.wall_distance:  # Only map close objects for clarity
                    angle = robot_theta + (i / num_points) * 2 * math.pi - math.pi
                    obs_x = robot_x + r * math.cos(angle)
                    obs_y = robot_y + r * math.sin(angle)
                    
                    # Convert to map coordinates
                    obs_mx, obs_my = self.world_to_map(obs_x, obs_y)
                    
                    if 0 <= obs_mx < self.map_size and 0 <= obs_my < self.map_size:
                        # Draw a small line segment to connect nearby points
                        # This creates sharper edges
                        if i > 0 and i < num_points - 1:
                            prev_r = ranges[i-1]
                            if prev_r < self.wall_distance:
                                prev_angle = robot_theta + ((i-1) / num_points) * 2 * math.pi - math.pi
                                prev_x = robot_x + prev_r * math.cos(prev_angle)
                                prev_y = robot_y + prev_r * math.sin(prev_angle)
                                prev_mx, prev_my = self.world_to_map(prev_x, prev_y)
                                
                                # Draw line between consecutive points
                                cv2.line(self.wall_map, (prev_mx, prev_my), (obs_mx, obs_my), 255, 2)
                        
                        # Mark individual obstacle points
                        self.obstacle_map[obs_my, obs_mx] = 255
    
    def process_lidar(self):
        """Fast LiDAR processing for navigation"""
        ranges = self.lidar.getRangeImage()
        robot_x, robot_y, robot_theta = self.get_robot_pose()
        
        # Update mapping every 2 steps for performance
        self.map_update_counter += 1
        if self.map_update_counter >= 2:
            self.fast_mapping(robot_x, robot_y, robot_theta, ranges)
            self.map_update_counter = 0
        
        # Check for front obstacles
        num_points = len(ranges)
        front_start = int(num_points * 0.375)  # Wider front cone (90 degrees)
        front_end = int(num_points * 0.625)
        
        front_ranges = ranges[front_start:front_end]
        if len(front_ranges) > 0:
            min_front = min(front_ranges)
            self.obstacle_detected = min_front < self.obstacle_threshold
            return min_front
        
        return float('inf')
        
    def rl_get_reward(self):
        """Calculate reward for RL agent"""
        reward = 0
        
        # Base reward for staying alive
        reward += 0.01
        
        # Reward for exploring new areas
        robot_x, robot_y, _ = self.get_robot_pose()
        cell_x = int((robot_x + 6) / 12 * 20)
        cell_y = int((robot_y + 5) / 10 * 20)
        
        if (cell_x, cell_y) not in self.rl_visited_cells:
            reward += 0.5  # Reward for exploring new cells
            self.rl_visited_cells.add((cell_x, cell_y))
        
        # Reward for detecting intruder
        if self.intruder_detected:
            reward += 2.0
        
        # Penalty for being near obstacles (but collision avoidance handles this)
        if self.obstacle_detected:
            reward -= 0.1
        
        # Penalty for staying in same area too long
        current_time = self.robot.getTime()
        if current_time - self.last_rl_update_time > 30:  # 30 seconds
            reward -= 0.2
        
        return reward
    
    def rl_choose_patrol_strategy(self, action):
        """Apply RL-chosen patrol strategy"""
        robot_x, robot_y, robot_theta = self.get_robot_pose()
        
        if action == 0:  # Explore Forward
            self.wander_angle = robot_theta
            self.robot_speed = 3.0
            
        elif action == 1:  # Turn Left
            self.wander_angle = robot_theta + math.pi / 4
            self.robot_speed = 2.5
            
        elif action == 2:  # Turn Right
            self.wander_angle = robot_theta - math.pi / 4
            self.robot_speed = 2.5
            
        elif action == 3:  # Move to Center
            # Calculate angle to center of room
            dx = 0 - robot_x
            dy = 0 - robot_y
            center_angle = math.atan2(dy, dx)
            self.wander_angle = center_angle
            self.robot_speed = 2.8
            
        elif action == 4:  # Check Corners
            # Cycle through corners
            corners = [(-5, -4), (-5, 4), (5, -4), (5, 4)]
            corner_idx = int(self.robot.getTime() / 10) % 4
            target_x, target_y = corners[corner_idx]
            dx = target_x - robot_x
            dy = target_y - robot_y
            corner_angle = math.atan2(dy, dx)
            self.wander_angle = corner_angle
            self.robot_speed = 3.0
            
        elif action == 5:  # Slow Patrol
            self.wander_angle = robot_theta + random.uniform(-0.3, 0.3)
            self.robot_speed = 2.0
        
        self.rl_action_duration = random.randint(50, 150)  # Action duration in steps
        self.last_rl_update_time = self.robot.getTime()
    
    def rl_update(self):
        """Update RL agent based on current state"""
        if self.intruder_detected:
            return  # RL doesn't control during intruder tracking
        
        current_time = self.robot.getTime()
        
        # Get current state
        robot_x, robot_y, robot_theta = self.get_robot_pose()
        state = self.rl_agent.discretize_state(
            robot_x, robot_y, robot_theta, 
            self.intruder_detected, 
            self.obstacle_detected
        )
        
        # Check if we need a new action
        if (self.current_rl_state is None or 
            self.rl_action_duration <= 0 or
            current_time - self.last_rl_update_time > 5):
            
            # If we had a previous state-action pair, update Q-value
            if self.current_rl_state is not None and self.current_rl_action is not None:
                reward = self.rl_get_reward()
                self.rl_agent.update_q_value(
                    self.current_rl_state, 
                    self.current_rl_action, 
                    reward, 
                    state
                )
                self.rl_rewards_history.append(reward)
                if len(self.rl_rewards_history) > 100:
                    self.rl_rewards_history.pop(0)
            
            # Choose new action
            self.current_rl_state = state
            self.current_rl_action = self.rl_agent.choose_action(state)
            self.rl_choose_patrol_strategy(self.current_rl_action)
            
            # Store Q-values for visualization
            self.rl_q_values = self.rl_agent.q_table[state].tolist()
            if len(self.rl_q_values) > 20:
                self.rl_q_values = self.rl_q_values[:20]
        
        else:
            self.rl_action_duration -= 1
        
        # Check if episode should end (e.g., after 2 minutes)
        if current_time - self.rl_episode_start_time > 120:  # 2 minutes
            self.rl_agent.end_episode()
            self.rl_episode_start_time = current_time
            self.rl_visited_cells.clear()