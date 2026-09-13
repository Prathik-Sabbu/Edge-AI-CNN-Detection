#include <Arduino.h>
#include <LiquidCrystal.h>
#include <NewPing.h>

// Initialize the LCD with our specific pins (RS, EN, D4, D5, D6, D7)
LiquidCrystal lcd(7, 6, 5, 4, 3, 2);

#define TRIGGER_PIN 11
#define ECHO_PIN 12
#define MAX_DISTANCE 40

NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);

// Timer variables for non-blocking cooldown
unsigned long lastTriggerTime = 0;
const unsigned long cooldown = 3000; // 3 seconds
bool isReady = true;

void setup() {
  Serial.begin(9600);

  // Set up the LCD's number of columns and rows
  lcd.begin(16, 2);
  lcd.print("System Ready...");
}

void loop() {
  // 1. LISTEN FOR PC: Check if the Python script sent a message
  if (Serial.available() > 0) {
    // Read the incoming text until a newline character is found
    String incomingMessage = Serial.readStringUntil('\n');
    incomingMessage.trim(); // Remove any invisible whitespace/carriage returns

    // Update the display
    lcd.clear();
    lcd.print(incomingMessage);
  }

  // 2. POLL SENSOR: Only ping if the 3-second cooldown has passed
  if (millis() - lastTriggerTime > cooldown) {

    // If we just finished cooling down, update the screen to "Ready" exactly
    // once
    if (!isReady) {
      lcd.clear();
      lcd.print("Ready...");
      isReady = true;
    }

    unsigned int distance = sonar.ping_cm();

    // If an object is in the goldilocks zone
    if (distance > 2) {
      Serial.println("TRIGGER"); // Sends to PC

      lcd.clear();
      lcd.print("TRIGGER!"); // Shows on LCD

      // Reset the timer so it waits 3 seconds before triggering again
      lastTriggerTime = millis();
      isReady = false; // Mark that we are now cooling down
    }
  }
}