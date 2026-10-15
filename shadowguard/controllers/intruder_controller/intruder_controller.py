"""
Intruder Robot Controller
Simple wandering behavior to test ShadowGuard detection
"""

from controller import Robot, Receiver
import random
import math

class IntruderController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Motors
        self.left_motor = self.robot.getDevice('intruder_left_motor')
        self.right_motor = self.robot.getDevice('intruder_right_motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Receiver for alerts (optional)
        self.receiver = self.robot.getDevice('receiver')
        if self.receiver:
            self.receiver.enable(self.timestep)
            self.receiver.setChannel(1)
        
        # Movement parameters
        self.base_speed = 2.0
        self.turn_duration = 0
        self.move_duration = 0
        self.current_state = 'wander'
        
        print("Intruder active - evading security...")
    
    def wander(self):
        """Random wandering behavior"""
        if self.move_duration <= 0:
            # Choose new action
            action = random.choice(['forward', 'turn_left', 'turn_right', 'pause'])
            
            if action == 'forward':
                self.left_motor.setVelocity(self.base_speed)
                self.right_motor.setVelocity(self.base_speed)
                self.move_duration = random.randint(20, 100)
            elif action == 'turn_left':
                self.left_motor.setVelocity(-self.base_speed)
                self.right_motor.setVelocity(self.base_speed)
                self.move_duration = random.randint(10, 30)
            elif action == 'turn_right':
                self.left_motor.setVelocity(self.base_speed)
                self.right_motor.setVelocity(-self.base_speed)
                self.move_duration = random.randint(10, 30)
            elif action == 'pause':
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)
                self.move_duration = random.randint(10, 40)
        else:
            self.move_duration -= 1
    
    def check_alerts(self):
        """Check if detected by security"""
        if self.receiver and self.receiver.getQueueLength() > 0:
            message = self.receiver.getString()
            if message.startswith("INTRUDER:"):
                print("Detected by security! Changing direction...")
                # Quick evasive turn
                self.left_motor.setVelocity(-self.base_speed * 1.5)
                self.right_motor.setVelocity(self.base_speed * 1.5)
                self.move_duration = 20
            self.receiver.nextPacket()
    
    def run(self):
        """Main control loop"""
        while self.robot.step(self.timestep) != -1:
            self.check_alerts()
            self.wander()

if __name__ == "__main__":
    controller = IntruderController()
    controller.run()