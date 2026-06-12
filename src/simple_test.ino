/*
  Simple ESP8266 Servo Test - No MQTT
  Just test WiFi and servo movement
*/

#include <ESP8266WiFi.h>
#include <Servo.h>

const char* WIFI_SSID = "EdNet";
const char* WIFI_PASSWORD = "Huawei@123";
const int SERVO_PIN = D4;

Servo myservo;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("=== ESP8266 Simple Test ===");
  
  // Test servo
  myservo.attach(SERVO_PIN);
  myservo.write(90);
  Serial.println("✓ Servo initialized at 90°");
  
  // Test WiFi
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int timeout = 20;
  while (WiFi.status() != WL_CONNECTED && timeout > 0) {
    delay(1000);
    Serial.print(".");
    timeout--;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi connected!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n✗ WiFi failed");
  }
  
  Serial.println("Starting servo sweep...");
}

void loop() {
  // Sweep servo
  for (int pos = 0; pos <= 180; pos += 10) {
    myservo.write(pos);
    Serial.print("Servo position: ");
    Serial.println(pos);
    delay(500);
  }
  
  for (int pos = 180; pos >= 0; pos -= 10) {
    myservo.write(pos);
    Serial.print("Servo position: ");
    Serial.println(pos);
    delay(500);
  }
}
