#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

Adafruit_MotorShield AFMS = Adafruit_MotorShield();
Adafruit_DCMotor *mot1 = AFMS.getMotor(1);
Adafruit_DCMotor *mot2 = AFMS.getMotor(2);
Adafruit_DCMotor *mot3 = AFMS.getMotor(3);


void setup(){
  Serial.begin(9600);
  AFMS.begin();
  mot1->setSpeed(0);
  mot2->setSpeed(0);
  mot3->setSpeed(0);
}

void loop() {
  if (Serial.available()>0){
    String command = Serial.readStringUntil('\n');
    int id = command.substring(0, command.indexOf(',')).toInt();
    int speed = command.substring(command.indexOf(',') + 1).toInt();
    switch(id){
      case 1:
        if(speed>=0){
          mot1->setSpeed(speed);
          mot1->run(FORWARD);
        }
        else{
          mot1->setSpeed(-speed);
          mot1->run(BACKWARD);
        }
        break;
      case 2:
        if(speed>=0){
          mot2->setSpeed(speed);
          mot2->run(FORWARD);
        }
        else{
          mot2->setSpeed(-speed);
          mot2->run(BACKWARD);
        }
        break;
      case 3:
        if(speed>=0){
          mot3->setSpeed(speed);
          mot3->run(FORWARD);
        }
        else{
          mot3->setSpeed(-speed);
          mot3->run(BACKWARD);
        }
        break;
      default :
        break;
    }
  }
}
