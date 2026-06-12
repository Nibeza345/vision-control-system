/*
  ESP8266 Single-Servo Controller for Face Tracking
  WIRING: Signal (yellow/orange) → D5 | Red → VIN | Brown → GND
*/

#include <ESP8266WiFi.h>
#define MQTT_MAX_PACKET_SIZE 512
#include <PubSubClient.h>
#include <Servo.h>
#include <ArduinoJson.h>

const char* MQTT_BROKER = "10.12.72.188";
const int MQTT_PORT = 1883;
const char* MQTT_TOPIC = "vision/nicole/movement";

const char* WIFI_SSID = "EdNet";
const char* WIFI_PASSWORD = "Huawei@123";

const int SERVO_PIN = D5;

const int MIN_MOVE_DELTA = 5;   // Ignore tiny angle changes (reduces jitter)
const int MAX_STEP = 2;         // Slow smooth follow
const int SEARCH_MAX_STEP = 6;
const unsigned long MOVE_INTERVAL_MS = 60;

Servo myServo;
int current_angle = 90;
int target_angle = 90;
bool search_mode = false;

WiFiClient espClient;
PubSubClient client(espClient);
unsigned long last_move_ms = 0;

void write_servo(int angle) {
  angle = constrain(angle, 0, 180);
  myServo.write(angle);
  current_angle = angle;
}

void boot_servo_test() {
  Serial.println("BOOT TEST: 45 -> 135 -> 90");
  int angles[] = {45, 135, 90};
  for (int i = 0; i < 3; i++) {
    write_servo(angles[i]);
    target_angle = angles[i];
    delay(700);
  }
  Serial.println("BOOT TEST done");
}

bool connect_wifi() {
  Serial.print("WiFi ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  for (int i = 0; i < 25 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500);
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

void connect_mqtt_once() {
  if (client.connected()) {
    return;
  }
  Serial.print("MQTT ");
  Serial.println(MQTT_BROKER);
  if (client.connect("esp8266_nicole")) {
    client.subscribe(MQTT_TOPIC);
    Serial.print("Subscribed ");
    Serial.println(MQTT_TOPIC);
  } else {
    Serial.print("MQTT fail ");
    Serial.println(client.state());
  }
}

int parse_angle(const char* msg, unsigned int len) {
  char buf[256];
  if (len >= sizeof(buf)) {
    len = sizeof(buf) - 1;
  }
  memcpy(buf, msg, len);
  buf[len] = '\0';

  StaticJsonDocument<128> doc;
  if (!deserializeJson(doc, buf)) {
    if (doc.containsKey("pan_angle")) {
      return doc["pan_angle"].as<int>();
    }
    if (doc.containsKey("servo_angle")) {
      return doc["servo_angle"].as<int>();
    }
  }

  const char* keys[] = {"\"pan_angle\"", "\"servo_angle\""};
  for (int k = 0; k < 2; k++) {
    const char* hit = strstr(buf, keys[k]);
    if (hit) {
      const char* colon = strchr(hit, ':');
      if (colon) {
        return atoi(colon + 1);
      }
    }
  }
  return -1;
}

void callback(char* topic, byte* payload, unsigned int length) {
  int angle = parse_angle((const char*)payload, length);
  if (angle < 0) {
    return;
  }

  String status = "TRACK";
  char buf[256];
  if (length < sizeof(buf)) {
    memcpy(buf, payload, length);
    buf[length] = '\0';
    StaticJsonDocument<96> doc;
    if (!deserializeJson(doc, buf) && doc.containsKey("status")) {
      status = doc["status"].as<String>();
    }
  }

  search_mode = (status == "SEARCH");
  target_angle = constrain(angle, 0, 180);

  Serial.print(search_mode ? "SEARCH " : "TRACK ");
  Serial.print("target=");
  Serial.println(target_angle);
}

void smooth_servo_tick() {
  if (millis() - last_move_ms < MOVE_INTERVAL_MS) {
    return;
  }

  int diff = target_angle - current_angle;
  if (abs(diff) < MIN_MOVE_DELTA) {
    return;
  }

  int limit = search_mode ? SEARCH_MAX_STEP : MAX_STEP;
  int step = diff;
  if (step > limit) {
    step = limit;
  } else if (step < -limit) {
    step = -limit;
  }

  write_servo(current_angle + step);
  last_move_ms = millis();
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n=== Face Lock Servo | D5 | team nicole ===");

  myServo.attach(SERVO_PIN, 500, 2500);
  write_servo(90);
  target_angle = 90;
  delay(400);
  boot_servo_test();

  if (!connect_wifi()) {
    ESP.restart();
  }

  client.setServer(MQTT_BROKER, MQTT_PORT);
  client.setCallback(callback);
  client.setBufferSize(512);
  connect_mqtt_once();

  Serial.println("Ready");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connect_wifi();
  }
  if (!client.connected()) {
    connect_mqtt_once();
    delay(100);
  }
  client.loop();
  smooth_servo_tick();
}
