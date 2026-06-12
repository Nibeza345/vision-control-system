/*
  ESP8266 Single-Servo Controller for Face Tracking
  Listens on MQTT and moves ONE servo (pan / horizontal).

  WIRING (3 wires):
    Yellow/Orange → D4 (signal)
    Red           → 5V
    Brown         → GND
*/

#include <ESP8266WiFi.h>
// face_locking.py JSON is ~400 bytes — default 256 truncates messages (servo never moves)
#define MQTT_MAX_PACKET_SIZE 1024
#include <PubSubClient.h>
#include <Servo.h>
#include <ArduinoJson.h>

// ----- CHANGE ONLY IF YOUR TEACHER GIVES DIFFERENT VALUES -----
const char* TEAM_ID = "nicole";           // Your project name (must match face_locking.py)
const char* MQTT_BROKER = "10.12.72.188"; // Your PC IP — run "ipconfig" if this changes
const int MQTT_PORT = 1883;
const char* MQTT_TOPIC = "vision/nicole/movement";

const char* WIFI_SSID = "EdNet";
const char* WIFI_PASSWORD = "Huawei@123";

const int SERVO_PIN = D4;  // Yellow wire goes here

Servo myServo;
int current_angle = 90;

WiFiClient espClient;
PubSubClient client(espClient);

bool connect_wifi() {
  Serial.print("WiFi ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
    delay(1000);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi OK");
    Serial.println(WiFi.localIP());
    return true;
  }

  Serial.println("\nWiFi FAILED");
  return false;
}

bool connect_mqtt() {
  while (!client.connected()) {
    Serial.print("MQTT ");
    Serial.println(MQTT_BROKER);
    if (client.connect("esp8266_nicole")) {
      client.subscribe(MQTT_TOPIC);
      Serial.print("Subscribed ");
      Serial.println(MQTT_TOPIC);
      return true;
    }
    Serial.print("MQTT fail ");
    Serial.println(client.state());
    delay(3000);
  }
  return true;
}

void move_servo(int target, bool fast) {
  target = constrain(target, 0, 180);
  int steps = fast ? 3 : 8;
  int step_delay = fast ? 8 : 15;
  for (int i = 0; i <= steps; i++) {
    int angle = current_angle + (target - current_angle) * i / steps;
    myServo.write(angle);
    delay(step_delay);
  }
  current_angle = target;
  Serial.print("Servo ");
  Serial.print(current_angle);
  Serial.println(fast ? " (search)" : "");
}

void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  DynamicJsonDocument doc(768);
  DeserializationError err = deserializeJson(doc, message);
  if (err) {
    Serial.print("JSON fail (len=");
    Serial.print(length);
    Serial.print("): ");
    Serial.println(err.c_str());
    return;
  }

  String status = doc["status"] | "UNKNOWN";
  int angle = doc["pan_angle"] | doc["servo_angle"] | current_angle;

  Serial.print("Got ");
  Serial.print(status);
  Serial.print(" angle=");
  Serial.println(angle);

  if (status == "STOP" || status == "NO_FACE") {
    move_servo(90, false);
  } else if (status == "SEARCH") {
    move_servo(angle, true);
  } else {
    move_servo(angle, false);
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("Face tracking servo — team nicole");

  myServo.attach(SERVO_PIN);
  myServo.write(90);

  if (!connect_wifi()) {
    ESP.restart();
  }

  client.setServer(MQTT_BROKER, MQTT_PORT);
  client.setCallback(callback);
  client.setBufferSize(1024);
  connect_mqtt();

  Serial.println("Ready.");
}

void loop() {
  if (!client.connected()) {
    connect_mqtt();
  }
  client.loop();
}
