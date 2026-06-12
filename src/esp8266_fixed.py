"""
ESP8266 Servo Controller for Face-Locked Tracking System
Receives MQTT movement commands and controls servo accordingly
FIXED VERSION - Corrected duty cycles
"""

import machine
import time
import network
import ujson
from machine import Pin, PWM
from umqtt.simple import MQTTClient

# ===================== CONFIGURATION =====================
TEAM_ID = "minister"  # Must match PC client team_id
MQTT_BROKER = "157.173.101.159"  # Replace with your VPS IP address
MQTT_PORT = 1883
MQTT_TOPIC = f"vision/{TEAM_ID}/movement"
MQTT_CLIENT_ID = f"esp8266_{TEAM_ID}"

# Servo configuration - CORRECTED VALUES
SERVO_PIN = 2  # GPIO pin for servo (D4 on NodeMCU)
SERVO_FREQ = 50  # 50Hz for standard servos
SERVO_MIN_DUTY = 26  # ~0.5ms pulse (0 degrees) - WORKING VALUES
SERVO_MAX_DUTY = 128  # ~2.5ms pulse (180 degrees) - WORKING VALUES
SERVO_CENTER_DUTY = 77  # ~1.5ms pulse (90 degrees) - WORKING VALUES

# WiFi configuration
WIFI_SSID = "RCA"  # Replace with your WiFi SSID
WIFI_PASSWORD = "@RcaNyabihu2023"  # Replace with your WiFi password

# ===================== GLOBAL VARIABLES =====================
servo = None
mqtt_client = None
current_angle = 90  # Start at center position
target_angle = 90

# ===================== SERVO FUNCTIONS =====================
def init_servo():
    """Initialize servo motor"""
    global servo
    try:
        servo = PWM(Pin(SERVO_PIN), SERVO_FREQ)
        servo.duty(SERVO_CENTER_DUTY)  # Start at center position
        print("✓ Servo initialized with corrected duty cycles")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize servo: {e}")
        return False

def angle_to_duty(angle):
    """Convert angle (0-180) to duty cycle"""
    duty = SERVO_MIN_DUTY + (angle / 180) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY)
    return int(duty)

def move_servo(angle):
    """Move servo to specified angle (0-180)"""
    global current_angle, target_angle
    try:
        # Clamp angle to valid range
        angle = max(0, min(180, angle))
        target_angle = angle

        # Smooth movement (gradual transition)
        steps = 10
        step_delay = 0.02  # 20ms between steps

        for i in range(steps + 1):
            intermediate_angle = current_angle + (target_angle - current_angle) * (i / steps)
            duty = angle_to_duty(intermediate_angle)
            servo.duty(duty)
            time.sleep(step_delay)

        current_angle = target_angle
        print(f"🔧 Servo moved to {angle}° (duty: {angle_to_duty(angle)})")

    except Exception as e:
        print(f"✗ Error moving servo: {e}")

# ===================== WIFI FUNCTIONS =====================
def connect_wifi():
    """Connect to WiFi network"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if not wlan.isconnected():
        print(f"Connecting to WiFi: {WIFI_SSID}")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)

        # Wait for connection
        timeout = 20
        while not wlan.isconnected() and timeout > 0:
            time.sleep(1)
            timeout -= 1
            print(".")

    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"✓ WiFi connected! IP: {ip}")
        return True
    else:
        print("✗ WiFi connection failed")
        return False

# ===================== MQTT FUNCTIONS =====================
def on_mqtt_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    if rc == 0:
        print(f"✓ Connected to MQTT broker at {MQTT_BROKER}")
        # Subscribe to movement topic
        client.subscribe(MQTT_TOPIC)
        print(f"✓ Subscribed to: {MQTT_TOPIC}")
    else:
        print(f"✗ MQTT connection failed: {rc}")

def on_mqtt_message(client, userdata, msg):
    """Handle incoming MQTT messages"""
    global current_angle

    try:
        # Parse JSON message
        payload = msg.payload.decode('utf-8')
        data = ujson.loads(payload)

        status = data.get('status', 'UNKNOWN')
        confidence = data.get('confidence', 0.0)
        servo_angle = data.get('servo_angle', None)
        direction = data.get('direction', 'UNKNOWN')
        degrees_from_center = data.get('degrees_from_center', 0)

        print(f"📡 Received: {status} | Confidence: {confidence:.2f} | Direction: {direction}")

        # Move servo based on command
        if servo_angle is not None:
            move_servo(servo_angle)
            print(f"📍 Face position: {degrees_from_center:.1f}° from center")

    except Exception as e:
        print(f"✗ Error processing MQTT message: {e}")

def init_mqtt():
    """Initialize MQTT client"""
    global mqtt_client
    try:
        mqtt_client = MQTTClient(
            MQTT_CLIENT_ID,
            MQTT_BROKER,
            MQTT_PORT,
            keepalive=60
        )

        mqtt_client.set_callback(on_mqtt_message)
        mqtt_client.set_callback(on_mqtt_connect)

        print(f"Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
        mqtt_client.connect()

        # Subscribe to movement topic
        mqtt_client.subscribe(MQTT_TOPIC)

        return True

    except Exception as e:
        print(f"✗ Failed to initialize MQTT: {e}")
        return False

# ===================== MAIN PROGRAM =====================
def main():
    """Main program loop"""
    print("=" * 50)
    print("ESP8266 Servo Controller - FIXED VERSION")
    print(f"Team ID: {TEAM_ID}")
    print("=" * 50)

    # Initialize components
    if not init_servo():
        print("Failed to initialize servo. Exiting.")
        return

    if not connect_wifi():
        print("Failed to connect to WiFi. Exiting.")
        return

    if not init_mqtt():
        print("Failed to initialize MQTT. Exiting.")
        return

    print("\n✓ System ready! Waiting for movement commands...")
    print("Press Ctrl+C to stop")

    # Main loop
    try:
        while True:
            # Check for MQTT messages
            mqtt_client.check_msg()
            time.sleep(0.1)  # Small delay to prevent overwhelming

    except KeyboardInterrupt:
        print("\n\nStopping program...")

    except Exception as e:
        print(f"\n✗ Error in main loop: {e}")

    finally:
        # Cleanup
        try:
            if mqtt_client:
                mqtt_client.disconnect()
                print("✓ MQTT disconnected")
        except:
            pass

        try:
            if servo:
                servo.duty(0)  # Turn off servo
                print("✓ Servo disabled")
        except:
            pass

        print("Program stopped")

if __name__ == "__main__":
    main()
