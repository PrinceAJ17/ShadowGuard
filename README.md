# ShadowGuard Security Robot with RL-Based Patrol

## Project Overview
ShadowGuard is an intelligent security robot system that combines Fast SLAM (Simultaneous Localization and Mapping) with Reinforcement Learning (Q-learning) for optimal autonomous patrolling. The system features:

* Real-time SLAM mapping with sharp edge detection
* Reinforcement Learning-based patrol optimization using Q-learning
* Intruder detection and tracking via RGB camera recognition
* Dual-robot simulation (security robot + intruder robot)
* Interactive visualization with OpenCV dashboards

## Key Features

### 1. Fast & Accurate SLAM Mapping
* High-resolution 800x800 mapping grid (2cm/pixel)
* LiDAR-based obstacle and wall detection
* Sharp edge detection using morphological operations
* Real-time trajectory and map visualization

### 2. Reinforcement Learning Patrol
* Q-learning agent with 4×4×4×2×2 state space (512 states)
* 6 distinct patrol strategies:
  1. Explore Forward
  2. Turn Left
  3. Turn Right
  4. Move to Center
  5. Check Corners
  6. Slow Patrol
* Automatic Q-table persistence (saves/loads to `q_table.json`)
* Epsilon-greedy exploration with decay

### 3. Intruder Detection & Response
* Color-based intruder recognition (red/orange objects)
* Real-time tracking with distance maintenance
* Alert system via Webots emitter
* Intruder trajectory visualization

### 4. Interactive Visualization
* **SLAM Map Window:** Shows real-time mapping, robot position, trajectory, and intruder locations
* **RL Dashboard Window:** Displays Q-values, reward history, exploration rate, and learning statistics
* Real-time status overlay with position, time, and patrol state

## System Requirements

### Hardware/Software
* Webots R2023b or later
* Python 3.7+
* Required Python packages:
  * `numpy`
  * `opencv-python (cv2)`

### Simulation Environment
* World Size: 12m × 10m house layout
* Furniture Objects: Sofa, tables, bed, wardrobe, bookshelf, kitchen counter
* Two Robots:
  * ShadowGuard Security Robot (blue)
  * Intruder Robot (red/orange)

## Installation Instructions

### 1. Download Webots
* Download and install Webots R2023b or later from [cyberbotics.com](https://cyberbotics.com)
* Webots includes Python support by default

### 2. Set Up Project
1. Create a new folder for your project
2. Copy the directory structure into it
3. Place the provided files in their respective folders

### 3. Install Python Dependencies
Open terminal/command prompt and run:
```bash
pip install numpy opencv-python
```

Or create a `requirements.txt`:
```txt
numpy>=1.21.0
opencv-python>=4.5.0
```

Install with:
```bash
pip install -r requirements.txt
```

### 4. Run the Simulation
1. Open Webots
2. File → Open World → Navigate to `worlds/house_patrol.wbt`
3. Click the "Play" button (▶) to start simulation
4. OpenCV windows will appear showing:
   * SLAM Map with robot position and mapping
   * RL Learning Dashboard (toggle with 'r' key)

## Quick Start Commands
```bash
# Clone or download the project
git clone <repository-url>
cd ShadowGuard-Security-Robot

# Install dependencies
pip install numpy opencv-python

# Open Webots and load the world file
# Or run from command line (if Webots is in PATH):
webots worlds/house_patrol.wbt
```