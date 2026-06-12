"""Test servo only — no camera. Run from src/ with venv active."""
import json
import time

import paho.mqtt.client as mqtt

MQTT_BROKER = "10.12.72.188"
MQTT_PORT = 1883
MQTT_TOPIC = "vision/nicole/movement"

# Big visible sweep
ANGLES = [30, 60, 90, 120, 150, 120, 90, 60, 30, 90]


def main():
    client = mqtt.Client()
    print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print(f"Publishing to {MQTT_TOPIC}")
    print("ESP Serial Monitor (115200): expect 'MQTT rx' and 'Servo ->'")
    print("On boot ESP also sweeps 45 -> 135 -> 90 automatically.\n")
    for angle in ANGLES:
        payload = json.dumps({"status": "SEARCH", "pan_angle": angle, "servo_angle": angle})
        client.publish(MQTT_TOPIC, payload)
        print(f"  sent {angle}°  ({len(payload)} bytes)")
        time.sleep(1.5)
    client.loop_stop()
    client.disconnect()
    print("\nDone. If servo did NOT move:")
    print("  1) Re-upload esp8266_servo_controller.ino (115200, COM5)")
    print("  2) Check boot sweep on power-up — if that fails, fix D5 + VIN + GND wiring")


if __name__ == "__main__":
    main()
