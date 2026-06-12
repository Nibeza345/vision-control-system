"""
WebSocket Backend API Service for Face-Locked Servo System
Relays MQTT messages to web dashboard via WebSocket
"""
import asyncio
import websockets
import json
import time
import paho.mqtt.client as mqtt
import threading
from typing import Set

# ===================== CONFIGURATION =====================
TEAM_ID = "sudoers"  # Your unique team identifier
MQTT_BROKER_HOST = "157.173.101.159"  # Your VPS MQTT broker
MQTT_BROKER_PORT = 1883
WEBSOCKET_PORT = 9002
MQTT_TOPIC = f"vision/{TEAM_ID}/movement"
MQTT_HEARTBEAT_TOPIC = f"vision/{TEAM_ID}/heartbeat"

# ===================== GLOBAL VARIABLES =====================
websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
mqtt_client = None
latest_movement_data = None
latest_heartbeat_data = {}

# ===================== MQTT FUNCTIONS =====================
def on_mqtt_connect(client, userdata, flags, rc):
    """Handle MQTT broker connection"""
    if rc == 0:
        print(f"✓ MQTT Broker connected on port {MQTT_BROKER_PORT}")
        # Subscribe to team topics
        client.subscribe(MQTT_TOPIC)
        client.subscribe(MQTT_HEARTBEAT_TOPIC)
        print(f"✓ Subscribed to: {MQTT_TOPIC}")
        print(f"✓ Subscribed to: {MQTT_HEARTBEAT_TOPIC}")
    else:
        print(f"✗ MQTT Broker connection failed: {rc}")

def on_mqtt_message(client, userdata, msg):
    """Handle incoming MQTT messages"""
    global latest_movement_data, latest_heartbeat_data

    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)

        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = int(time.time())

        # Store latest data
        if topic == MQTT_TOPIC:
            latest_movement_data = data
            print(f"📡 MQTT Movement: {data.get('status', 'UNKNOWN')} | Angle: {data.get('servo_angle', 'N/A')}°")
        elif topic == MQTT_HEARTBEAT_TOPIC:
            node = data.get('node', 'unknown')
            latest_heartbeat_data[node] = data
            print(f"💓 MQTT Heartbeat from {node}: {data.get('status', 'UNKNOWN')}")

        # Broadcast to all WebSocket clients
        message = {
            "type": "mqtt_message",
            "topic": topic,
            "data": data
        }

        asyncio.run(broadcast_to_clients(json.dumps(message)))

    except Exception as e:
        print(f"✗ Error processing MQTT message: {e}")

def init_mqtt_broker():
    """Initialize MQTT broker client"""
    global mqtt_client

    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message

        print(f"Connecting to MQTT broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
        mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)

        # Start MQTT loop in separate thread
        mqtt_thread = threading.Thread(target=mqtt_client.loop_forever, daemon=True)
        mqtt_thread.start()

        return True

    except Exception as e:
        print(f"✗ Failed to initialize MQTT broker: {e}")
        return False

# ===================== WEBSOCKET FUNCTIONS =====================
async def broadcast_to_clients(message):
    """Broadcast message to all connected WebSocket clients"""
    if websocket_clients:
        # Create a list of disconnected clients to remove
        disconnected = []

        for client in websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
            except Exception as e:
                print(f"✗ Error sending to client: {e}")
                disconnected.append(client)

        # Remove disconnected clients
        for client in disconnected:
            websocket_clients.discard(client)

async def handle_websocket_client(websocket, path):
    """Handle new WebSocket client connection"""
    client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    print(f"🔗 WebSocket client connected: {client_addr}")

    # Add client to set
    websocket_clients.add(websocket)

    try:
        # Send latest data immediately upon connection
        welcome_message = {
            "type": "welcome",
            "team_id": TEAM_ID,
            "timestamp": int(time.time()),
            "message": f"Connected to {TEAM_ID} team dashboard"
        }
        await websocket.send(json.dumps(welcome_message))

        # Send latest movement data if available
        if latest_movement_data:
            movement_message = {
                "type": "mqtt_message",
                "topic": MQTT_TOPIC,
                "data": latest_movement_data
            }
            await websocket.send(json.dumps(movement_message))

        # Send latest heartbeat data if available
        if latest_heartbeat_data:
            heartbeat_message = {
                "type": "mqtt_message",
                "topic": MQTT_HEARTBEAT_TOPIC,
                "data": latest_heartbeat_data
            }
            await websocket.send(json.dumps(heartbeat_message))

        # Keep connection alive and handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"📨 Received from {client_addr}: {data}")

                # Handle client requests (e.g., ping)
                if data.get('type') == 'ping':
                    pong_response = {
                        "type": "pong",
                        "timestamp": int(time.time())
                    }
                    await websocket.send(json.dumps(pong_response))

            except json.JSONDecodeError:
                print(f"✗ Invalid JSON from {client_addr}: {message}")
            except Exception as e:
                print(f"✗ Error handling message from {client_addr}: {e}")

    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 WebSocket client disconnected: {client_addr}")
    except Exception as e:
        print(f"✗ Error with client {client_addr}: {e}")
    finally:
        # Remove client from set
        websocket_clients.discard(websocket)
        print(f"🧹 Cleaned up client: {client_addr}")

async def start_websocket_server():
    """Start WebSocket server"""
    print(f"🚀 Starting WebSocket server on port {WEBSOCKET_PORT}")

    try:
        # Start WebSocket server
        server = await websockets.serve(
            handle_websocket_client,
            "0.0.0.0",  # Listen on all interfaces
            WEBSOCKET_PORT
        )

        print(f"✓ WebSocket server listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
        print(f"✓ Dashboard URL: ws://YOUR_VPS_IP:{WEBSOCKET_PORT}")

        return server

    except Exception as e:
        print(f"✗ Failed to start WebSocket server: {e}")
        return None

# ===================== MAIN PROGRAM =====================
async def main():
    """Main program"""
    print("=" * 60)
    print("WebSocket Backend API Service")
    print(f"Team ID: {TEAM_ID}")
    print(f"MQTT Broker Port: {MQTT_BROKER_PORT}")
    print(f"WebSocket Port: {WEBSOCKET_PORT}")
    print("=" * 60)

    # Initialize MQTT broker connection
    if not init_mqtt_broker():
        print("Failed to initialize MQTT broker. Exiting.")
        return

    # Start WebSocket server
    server = await start_websocket_server()
    if not server:
        print("Failed to start WebSocket server. Exiting.")
        return

    print("\n✓ Backend API service is running!")
    print("📡 MQTT messages will be relayed to WebSocket clients")
    print("🌐 Web dashboard can connect via WebSocket")
    print("\nPress Ctrl+C to stop")

    try:
        # Keep server running
        await server.wait_closed()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Cleanup
        if mqtt_client:
            mqtt_client.disconnect()
            print("✓ MQTT disconnected")

        print("✓ Backend service stopped")

if __name__ == "__main__":
    asyncio.run(main())
