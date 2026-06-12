"""
Test D4 pin by blinking built-in LED
D4 on NodeMCU is also the built-in LED
"""

import machine
import time

led = machine.Pin(2, machine.Pin.OUT)  # GPIO2 = D4 = built-in LED

print("Testing D4 pin with LED...")

for i in range(10):
    led.on()
    print("LED ON")
    time.sleep(1)
    led.off() 
    print("LED OFF")
    time.sleep(1)

print("LED test complete!")
