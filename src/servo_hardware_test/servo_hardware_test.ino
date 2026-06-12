/*
  HARDWARE-ONLY servo test — no WiFi, no MQTT.
  Upload this alone to prove the servo + wiring work.

  Yellow/orange -> D5
  Red           -> VIN
  Brown         -> GND
*/

#include <Servo.h>

const int SERVO_PIN = D5;

Servo servo;

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("=== SERVO HARDWARE TEST ===");
  Serial.println("Signal=D5  Power=VIN  GND=GND");
  Serial.println("Watch the horn: 0 -> 180 -> 90 every 3 seconds");
  Serial.println();

  // Wide pulse range helps cheap SG90-style servos
  servo.attach(SERVO_PIN, 500, 2500);
  servo.write(90);
  delay(500);
}

void loop() {
  int positions[] = {0, 90, 180, 90};
  for (int i = 0; i < 4; i++) {
    int angle = positions[i];
    servo.write(angle);
    Serial.print("angle ");
    Serial.println(angle);
    delay(2000);
  }
}
