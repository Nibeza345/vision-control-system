"""
Simple Servo Test - Move servo through full range
"""
import machine
import time
from machine import Pin, PWM

# Servo configuration
SERVO_PIN = 2  # GPIO pin for servo (D4 on NodeMCU)
SERVO_FREQ = 50  # 50Hz for standard servos
SERVO_MIN_DUTY = 40  # ~0.5ms pulse (0 degrees)
SERVO_MAX_DUTY = 115  # ~2.5ms pulse (180 degrees)
SERVO_CENTER_DUTY = 77  # ~1.5ms pulse (90 degrees)

def angle_to_duty(angle):
    """Convert angle (0-180) to duty cycle"""
    duty = SERVO_MIN_DUTY + (angle / 180) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY)
    return int(duty)

print("Testing servo motor...")
servo = PWM(Pin(SERVO_PIN), SERVO_FREQ)

# Test sequence
angles = [90, 0, 180, 90, 45, 135, 90]

for angle in angles:
    duty = angle_to_duty(angle)
    servo.duty(duty)
    print(f"Moving to {angle}° (duty: {duty})")
    time.sleep(2)

servo.duty(0)  # Turn off servo
print("Test complete!")
