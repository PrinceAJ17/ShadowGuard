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
    def update_display_map(self):
        """Update the display map with sharp edges using morphological operations"""
        with self.map_lock:
            # Start with light gray background
            self.display_map.fill(220)
            
            # Process wall map to get sharp edges
            if np.any(self.wall_map > 0):
                # Apply threshold
                _, binary_walls = cv2.threshold(self.wall_map, 50, 255, cv2.THRESH_BINARY)
                
                # Apply morphological operations for cleaner edges
                kernel = np.ones((3, 3), np.uint8)
                
                # Close small gaps
                closed = cv2.morphologyEx(binary_walls, cv2.MORPH_CLOSE, kernel)
                
                # Find edges using Canny
                edges = cv2.Canny(closed, 50, 150)
                
                # Dilate edges slightly for visibility
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                # Draw walls in blue
                wall_mask = closed > 0
                self.display_map[wall_mask] = [180, 180, 255]  # Light blue
                
                # Draw edges in dark blue
                edge_mask = edges > 0
                self.display_map[edge_mask] = [100, 100, 200]  # Dark blue
            
            # Process obstacle map for furniture/boxes
            if np.any(self.obstacle_map > 0):
                _, binary_obstacles = cv2.threshold(self.obstacle_map, 50, 255, cv2.THRESH_BINARY)
                
                # Apply morphological operations
                kernel = np.ones((3, 3), np.uint8)
                closed_obs = cv2.morphologyEx(binary_obstacles, cv2.MORPH_CLOSE, kernel)
                
                # Find contours for filled objects
                contours, _ = cv2.findContours(closed_obs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw filled contours for furniture
                for contour in contours:
                    if cv2.contourArea(contour) > 10:  # Filter small noise
                        # Create mask for this object
                        obj_mask = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
                        cv2.drawContours(obj_mask, [contour], -1, 255, -1)
                        
                        # Draw object in brown
                        obj_mask_bool = obj_mask > 0
                        self.display_map[obj_mask_bool] = [150, 120, 90]  # Brown
                        
                        # Draw outline in dark brown
                        cv2.drawContours(self.display_map, [contour], -1, [100, 80, 60], 2)
    
    def autonomous_patrol(self):
        """Fast autonomous patrolling with obstacle avoidance and RL integration"""
        robot_x, robot_y, robot_theta = self.get_robot_pose()
        
        if self.obstacle_detected:
            self.patrol_state = 'avoiding'
            self.avoid_timer = 15
            
            # Move backward and turn (override RL during obstacle avoidance)
            self.left_motor.setVelocity(-2.5)
            self.right_motor.setVelocity(-1.5)
        elif self.patrol_state == 'avoiding' and self.avoid_timer > 0:
            self.avoid_timer -= 1
            if self.avoid_timer <= 8:
                # Sharp turn
                self.left_motor.setVelocity(3.0)
                self.right_motor.setVelocity(-3.0)
        elif self.patrol_state == 'avoiding':
            # Resume exploration
            self.patrol_state = 'exploring'
            # Let RL choose new strategy after avoidance
            self.current_rl_state = None
        else:
            # Normal exploration with RL guidance
            self.patrol_state = 'exploring'
            
            # Update RL agent
            self.rl_update()
            
            # Simple wander behavior with RL-adjusted parameters
            base_speed = self.robot_speed
            angle_diff = self.wander_angle - robot_theta
            
            # Normalize angle
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            turn_gain = 0.8
            angular_speed = turn_gain * angle_diff
            
            left_speed = base_speed - angular_speed
            right_speed = base_speed + angular_speed
            
            # Clamp speeds
            max_speed = 3.5
            left_speed = max(-max_speed, min(max_speed, left_speed))
            right_speed = max(-max_speed, min(max_speed, right_speed))
            
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
            
            # Add trajectory point
            with self.trajectory_lock:
                self.trajectory.append((robot_x, robot_y))
                if len(self.trajectory) > 500:
                    self.trajectory.pop(0)

    def detect_intruder(self):
        """Fast intruder detection"""
        num_objects = self.camera.getRecognitionNumberOfObjects()
        
        if num_objects > 0:
            objects = self.camera.getRecognitionObjects()
            for obj in objects:
                colors = obj.getColors()
                try:
                    r, g, b = colors[0], colors[1], colors[2]
                    # Detect red/orange intruder
                    if r > 0.6 and g < 0.4 and b < 0.4:
                        position = obj.getPosition()
                        robot_x, robot_y, robot_theta = self.get_robot_pose()
                        
                        # Convert to world coordinates
                        local_x = position[0]
                        local_y = position[1]
                        
                        intruder_x = robot_x + local_x * math.cos(robot_theta) - local_y * math.sin(robot_theta)
                        intruder_y = robot_y + local_x * math.sin(robot_theta) + local_y * math.cos(robot_theta)
                        
                        with self.intruder_data_lock:
                            first_sighting = not self.intruder_detected
                            self.intruder_detected = True
                            self.intruder_position = (intruder_x, intruder_y)
                            self.intruder_trajectory.append((intruder_x, intruder_y))
                            if len(self.intruder_trajectory) > 100:
                                self.intruder_trajectory.pop(0)
                        
                        # Alert if first detection
                        if self.emitter and first_sighting:
                            alert_msg = f"INTRUDER:{intruder_x:.1f}:{intruder_y:.1f}"
                            self.emitter.send(alert_msg.encode('utf-8'))
                            print(f"🚨 INTRUDER at ({intruder_x:.1f}, {intruder_y:.1f})")
                        
                        return True
                except:
                    continue
        
        # Reset if no longer detected
        with self.intruder_data_lock:
            self.intruder_detected = False
        
        return False
    
    def track_intruder(self):
        """Track intruder while maintaining distance"""
        with self.intruder_data_lock:
            if not self.intruder_position or not self.intruder_detected:
                return
        
        target_x, target_y = self.intruder_position
        robot_x, robot_y, robot_theta = self.get_robot_pose()
        
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = target_angle - robot_theta
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Maintain 1.5m distance
        desired_distance = 1.5
        
        if distance > desired_distance + 0.3:
            # Approach
            forward_speed = min(2.5, (distance - desired_distance) * 1.0)
        elif distance < desired_distance - 0.3:
            # Back up
            forward_speed = -min(2.0, (desired_distance - distance) * 1.0)
        else:
            # Maintain distance
            forward_speed = 0
        
        # Add turning to face intruder
        angular_speed = 1.5 * angle_diff
        
        left_speed = forward_speed - angular_speed
        right_speed = forward_speed + angular_speed
        
        # Clamp speeds
        max_speed = 3.0
        left_speed = max(-max_speed, min(max_speed, left_speed))
        right_speed = max(-max_speed, min(max_speed, right_speed))
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def display_rl_dashboard(self):
        """Display RL learning dashboard in separate window"""
        dashboard = np.ones((600, 800, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Title
        cv2.putText(dashboard, "ShadowGuard RL Learning Dashboard", (50, 50),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
        
        # RL Statistics
        stats_y = 100
        cv2.putText(dashboard, f"Episode: {self.rl_agent.episode_count}", (50, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(dashboard, f"Total Reward: {self.rl_agent.total_rewards:.1f}", (50, stats_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(dashboard, f"Exploration Rate (ε): {self.rl_agent.epsilon:.3f}", (50, stats_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(dashboard, f"Current Action: {self.rl_agent.action_meanings[self.current_rl_action] if self.current_rl_action is not None else 'None'}", 
                   (50, stats_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 200), 2)
        
        # Q-values bar chart
        if self.rl_q_values:
            chart_y = 200
            chart_height = 150
            chart_width = 600
            max_q = max(abs(min(self.rl_q_values)), abs(max(self.rl_q_values))) + 0.1
            
            
            
            for i, q_val in enumerate(self.rl_q_values):
                bar_width = 30
                bar_x = 50 + i * 40
                bar_height = int((q_val / max_q) * chart_height / 2)
                
                # Color based on value
                if q_val >= 0:
                    color = (0, int(200 * (q_val/max_q)), 0)  # Green for positive
                else:
                    color = (0, 0, int(200 * (-q_val/max_q)))  # Red for negative
                
                # Draw bar
                bar_y = chart_y + chart_height//2
                cv2.rectangle(dashboard, 
                            (bar_x, bar_y - bar_height),
                            (bar_x + bar_width, bar_y),
                            color, -1)
                
                # Draw border
                cv2.rectangle(dashboard, 
                            (bar_x, bar_y - bar_height),
                            (bar_x + bar_width, bar_y),
                            (50, 50, 50), 1)
                
                # Label
                cv2.putText(dashboard, f"{i}", (bar_x + 5, bar_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(dashboard, f"{q_val:.2f}", (bar_x, bar_y - bar_height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Reward history graph
        if self.rl_rewards_history:
            graph_y = 400
            graph_height = 150
            graph_width = 600
            
            cv2.putText(dashboard, "Reward History (Last 100 steps):", (50, graph_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Draw grid
            for i in range(0, 11):
                y = graph_y + int(i * graph_height / 10)
                cv2.line(dashboard, (50, y), (50 + graph_width, y), (200, 200, 200), 1)
            
            # Draw reward line
            points = []
            max_reward = max(self.rl_rewards_history) if self.rl_rewards_history else 1
            min_reward = min(self.rl_rewards_history) if self.rl_rewards_history else 0
            
            for i, reward in enumerate(self.rl_rewards_history):
                x = 50 + int(i * graph_width / len(self.rl_rewards_history))
                if max_reward != min_reward:
                    y = graph_y + graph_height - int((reward - min_reward) / (max_reward - min_reward) * graph_height)
                else:
                    y = graph_y + graph_height // 2
                points.append((x, y))
            
            for i in range(1, len(points)):
                cv2.line(dashboard, points[i-1], points[i], (0, 100, 200), 2)
            
            # Draw zero line
            if min_reward < 0 < max_reward:
                zero_y = graph_y + graph_height - int((0 - min_reward) / (max_reward - min_reward) * graph_height)
                cv2.line(dashboard, (50, zero_y), (50 + graph_width, zero_y), (100, 100, 100), 1)
        
        # Legend
        legend_y = 570
        cv2.putText(dashboard, "Action Legend:", (50, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        for i, action_name in enumerate(self.rl_agent.action_meanings):
            if i < 3:  # First column
                cv2.putText(dashboard, f"{i}: {action_name}", (50, legend_y + 25 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:  # Second column
                cv2.putText(dashboard, f"{i}: {action_name}", (300, legend_y + 25 + (i-3)*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Status
        status = "LEARNING" if not self.intruder_detected else "TRACKING - RL PAUSED"
        
        
        cv2.imshow(self.rl_window_name, dashboard)
        
    def display_map_thread(self):
        """Display thread for OpenCV windows - Thread-safe version"""
        cv2.namedWindow(self.map_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.map_window_name, 700, 700)
        
        if self.show_rl:
            cv2.namedWindow(self.rl_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.rl_window_name, 800, 600)
        
        while self.show_map:
            try:
                # Update display map
                self.update_display_map()
                
                # Create a copy of the display map to draw on
                with self.map_lock:
                    display_copy = self.display_map.copy()
                
                # Get thread-safe copies of trajectory data
                with self.trajectory_lock:
                    trajectory_copy = list(self.trajectory)
                
                with self.intruder_data_lock:
                    intruder_detected = self.intruder_detected
                    intruder_position = self.intruder_position
                    intruder_trajectory_copy = list(self.intruder_trajectory)
                
                # Draw RL visited cells
                for cell_x, cell_y in self.rl_visited_cells:
                    map_x = int(self.map_center + (cell_x - 10) * (12/20) / self.map_resolution)
                    map_y = int(self.map_center - (cell_y - 10) * (10/20) / self.map_resolution)
                    size = int(0.5 / self.map_resolution)
                    cv2.rectangle(display_copy, 
                                (map_x - size//2, map_y - size//2),
                                (map_x + size//2, map_y + size//2),
                                (200, 255, 200), 1)
                
                # Draw robot trajectory (green)
                if len(trajectory_copy) > 1:
                    points = []
                    for pos in trajectory_copy:
                        mx, my = self.world_to_map(pos[0], pos[1])
                        points.append((mx, my))
                    
                    # Draw as connected line
                    for i in range(len(points) - 1):
                        cv2.line(display_copy, points[i], points[i+1], (0, 200, 0), 2)
                
                # Draw robot position
                robot_x, robot_y, robot_theta = self.get_robot_pose()
                robot_mx, robot_my = self.world_to_map(robot_x, robot_y)
                
                # Draw robot as green circle with direction
                cv2.circle(display_copy, (robot_mx, robot_my), 8, (0, 255, 0), -1)
                cv2.circle(display_copy, (robot_mx, robot_my), 8, (0, 100, 0), 2)  # Outline
                
                # Direction arrow
                arrow_len = 15
                end_x = int(robot_mx + arrow_len * math.cos(robot_theta))
                end_y = int(robot_my - arrow_len * math.sin(robot_theta))
                cv2.arrowedLine(display_copy, (robot_mx, robot_my), (end_x, end_y), 
                              (0, 150, 0), 2, tipLength=0.3)
                
                # Draw RL action indicator
                if self.current_rl_action is not None and not intruder_detected:
                    action_text = f"RL: {self.rl_agent.action_meanings[self.current_rl_action]}"
                    cv2.putText(display_copy, action_text, (robot_mx + 20, robot_my - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 200), 2)
                
                # Draw intruder if detected
                if intruder_detected and intruder_position:
                    # Draw trajectory (orange)
                    if len(intruder_trajectory_copy) > 1:
                        for i in range(len(intruder_trajectory_copy) - 1):
                            x1, y1 = self.world_to_map(intruder_trajectory_copy[i][0], 
                                                      intruder_trajectory_copy[i][1])
                            x2, y2 = self.world_to_map(intruder_trajectory_copy[i+1][0], 
                                                      intruder_trajectory_copy[i+1][1])
                            cv2.line(display_copy, (x1, y1), (x2, y2), (0, 100, 255), 2)
                    
                    # Draw current position
                    intr_mx, intr_my = self.world_to_map(intruder_position[0], 
                                                        intruder_position[1])
                    
                    # Red circle for intruder
                    cv2.circle(display_copy, (intr_mx, intr_my), 10, (0, 0, 255), -1)
                    cv2.circle(display_copy, (intr_mx, intr_my), 10, (0, 0, 100), 2)  # Outline
                    
                    cv2.putText(display_copy, "INTRUDER", (intr_mx + 12, intr_my - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Add grid lines for scale (1 meter)
                grid_spacing = int(1.0 / self.map_resolution)
                grid_color = (200, 200, 200)
                for i in range(0, self.map_size, grid_spacing):
                    cv2.line(display_copy, (i, 0), (i, self.map_size), grid_color, 1)
                    cv2.line(display_copy, (0, i), (self.map_size, i), grid_color, 1)
                
                # Add status overlay
                overlay = display_copy.copy()
                cv2.rectangle(overlay, (5, 5), (350, 120), (255, 255, 255), -1)
                cv2.addWeighted(overlay, 0.7, display_copy, 0.3, 0, display_copy)
                
                # Status text
                status = "TRACKING INTRUDER" if intruder_detected else f"PATROL: {self.patrol_state.upper()}"
                cv2.putText(display_copy, f"Status: {status}", (10, 30),
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
                cv2.putText(display_copy, f"Time: {self.robot.getTime():.1f}s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(display_copy, f"Position: ({robot_x:.1f}, {robot_y:.1f})", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(display_copy, f"RL Episode: {self.rl_agent.episode_count}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 1)
                
                # Scale bar
                scale_m = 5
                scale_px = int(scale_m / self.map_resolution)
                bar_x = self.map_size - 150
                bar_y = self.map_size - 30
                cv2.line(display_copy, (bar_x, bar_y), (bar_x + scale_px, bar_y), (0, 0, 0), 3)
                cv2.putText(display_copy, f"{scale_m}m", (bar_x + scale_px//2 - 15, bar_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # Legend
                cv2.putText(display_copy, "Legend:", (self.map_size - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.rectangle(display_copy, (self.map_size - 150, 40), (self.map_size - 130, 55), 
                            (180, 180, 255), -1)  # Wall
                cv2.putText(display_copy, "Walls", (self.map_size - 125, 52),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.rectangle(display_copy, (self.map_size - 150, 60), (self.map_size - 130, 75),
                            (150, 120, 90), -1)  # Furniture
                cv2.putText(display_copy, "Furniture", (self.map_size - 125, 72),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.rectangle(display_copy, (self.map_size - 150, 80), (self.map_size - 130, 95),
                            (200, 255, 200), -1)  # RL visited
                cv2.putText(display_copy, "RL Visited", (self.map_size - 125, 92),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Show map
                cv2.imshow(self.map_window_name, display_copy)
                
                # Show RL dashboard if enabled
                if self.show_rl:
                    self.display_rl_dashboard()
                
                # Check for ESC key
                key = cv2.waitKey(50) & 0xFF
                if key == 27:  # ESC
                    self.show_map = False
                    self.show_rl = False
                    break
                elif key == ord('r'):  # Toggle RL dashboard
                    self.show_rl = not self.show_rl
                    if not self.show_rl:
                        cv2.destroyWindow(self.rl_window_name)
                    
            except Exception as e:
                print(f"Display error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    def update_camera_display(self):
        """Update Webots camera display"""
        img = self.camera.getImage()
        if img:
            self.camera_display.imageNew(img, Display.BGRA, 
                                        self.camera.getWidth(), 
                                        self.camera.getHeight())
            
            if self.intruder_detected:
                self.camera_display.setColor(0xFF0000)
                w = self.camera_display.getWidth()
                h = self.camera_display.getHeight()
                for i in range(3):
                    self.camera_display.drawRectangle(i, i, w-2*i, h-2*i)

    def run(self):
        """Main control loop"""
        print("\nShadowGuard Fast SLAM with RL Online")
        print("Starting autonomous patrol with Q-learning...")
        print("Press ESC in any window to close")
        print("Press 'r' to toggle RL dashboard")
        
        # Initialize RL episode
        self.rl_episode_start_time = self.robot.getTime()
        
        # Start display thread
        display_thread = threading.Thread(target=self.display_map_thread, daemon=True)
        display_thread.start()
        
        while self.robot.step(self.timestep) != -1:
            # Process LiDAR (fast)
            min_dist = self.process_lidar()
            
            # Detect intruder
            intruder_seen = self.detect_intruder()
            
            # Control behavior
            if self.intruder_detected:
                self.track_intruder()
            else:
                self.autonomous_patrol()
            
            # Update camera display
            self.update_camera_display()
            
            # Status output (less frequent)
            if int(self.robot.getTime() * 10) % 150 == 0:
                robot_x, robot_y, robot_theta = self.get_robot_pose()
                status = "TRACKING" if self.intruder_detected else "PATROL"
                rl_info = f"RL Action: {self.rl_agent.action_meanings[self.current_rl_action]}" if self.current_rl_action is not None else "RL: Choosing"
                print(f"[{self.robot.getTime():.1f}s] {status} | Pos: ({robot_x:.1f}, {robot_y:.1f}) | {rl_info}")
        
        # Cleanup
        self.show_map = False
        self.show_rl = False
        
        # Save final Q-table
        self.rl_agent.save_q_table()
        print("Simulation ended. RL Q-table saved.")

# Run controller
if __name__ == "__main__":
    controller = ShadowGuardSLAM()
    controller.run()