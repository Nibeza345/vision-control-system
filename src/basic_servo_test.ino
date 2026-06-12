/*
  ESP8266 NodeMCU Servo Test
  Standard servo library test
*/

// Use ESP8266 specific servo approach
#define SERVO_PIN D4  // GPIO2 on NodeMCU

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Configure servo pin
  pinMode(SERVO_PIN, OUTPUT);
  Serial.println("Starting servo test without library...");
  
  // Test basic PWM signal
  for(int i = 0; i < 10; i++) {
    Serial.println("Testing servo pulse...");
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(1500);  // 1.5ms pulse (center)
    digitalWrite(SERVO_PIN, LOW);
    delay(20);  // 50Hz refresh rate
  }
  
  Serial.println("Basic test complete!");
}

void loop() {
  // Simple sweep without Servo library
  for(int pulse = 500; pulse <= 2500; pulse += 100) {
    Serial.print("Pulse width: ");
    Serial.print(pulse);
    Serial.println(" microseconds");
    
    for(int i = 0; i < 50; i++) {
      digitalWrite(SERVO_PIN, HIGH);
      delayMicroseconds(pulse);
      digitalWrite(SERVO_PIN, LOW);
      delay(20);
    }
    delay(500);
  }
}
