/*
  Simple Servo Test for NodeMCU
  Move servo through full range to test hardware
*/

#include <Servo.h>

Servo myservo;  // Create servo object

int servo_pin = D4;  // GPIO2 on NodeMCU

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("Starting servo test...");
  
  myservo.attach(servo_pin);  // Attach servo to pin D4
  
  // Test sequence
  Serial.println("Moving to 90 degrees (center)");
  myservo.write(90);
  delay(2000);
  
  Serial.println("Moving to 0 degrees");
  myservo.write(0);
  delay(2000);
  
  Serial.println("Moving to 180 degrees");
  myservo.write(180);
  delay(2000);
  
  Serial.println("Moving back to 90 degrees");
  myservo.write(90);
  delay(2000);
  
  Serial.println("Test complete!");
}

void loop() {
  // Sweep servo continuously
  for (int pos = 0; pos <= 180; pos += 5) {
    myservo.write(pos);
    Serial.print("Position: ");
    Serial.println(pos);
    delay(100);
  }
  for (int pos = 180; pos >= 0; pos -= 5) {
    myservo.write(pos);
    Serial.print("Position: ");
    Serial.println(pos);
    delay(100);
  }
}
