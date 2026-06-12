"""Sweep the servo via MQTT — no camera needed. Run from src/ with venv active."""
import json
import time

import paho.mqtt.client as mqtt

MQTT_BROKER = "10.12.72.188"
MQTT_PORT = 1883
MQTT_TOPIC = "vision/nicole/movement"

ANGLES = [60, 90, 120, 90, 60, 90]


def main():
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print(f"Publishing to {MQTT_TOPIC} on {MQTT_BROKER}:{MQTT_PORT}")
    print("Watch ESP Serial Monitor (115200) for 'Got ... angle=...' and 'Servo ...'")
    for angle in ANGLES:
        msg = {
            "status": "MOVE_RIGHT" if angle > 90 else "MOVE_LEFT",
            "pan_angle": angle,
            "servo_angle": angle,
        }
        client.publish(MQTT_TOPIC, json.dumps(msg))
        print(f"  -> {angle}°")
        time.sleep(2)
    client.loop_stop()
    client.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
