#! /usr/bin/env python3
# =========================================================
# file name:    detect_contact.py
# description:  detect contact between simulated patient and probe
# author:       Xihan Ma
# date:         2024-01-30
# =========================================================

import numpy as np

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3

class DetectContact():

  def __init__(self) -> None:
    self.isContact = False
    self.nonContactCounter = 0
    self.contact_norm = np.zeros(3)
    self.contact_pos = np.zeros(3)

    rospy.init_node('detect_contact', anonymous=True)
    rospy.Subscriber('/patient/contact/position', Vector3, self.patient_contact_pos_cb)
    rospy.Subscriber('/patient/contact/normal', Vector3, self.patient_contact_norm_cb)
    self.contact_state_pub = rospy.Publisher('/patient/contact/isContact', Bool, queue_size=1)
    self.rate = rospy.Rate(100)

  def patient_contact_pos_cb(self, msg: Vector3):
    # if msg.x != -1 and msg.y != -1 and msg.z != -1:
    self.contact_pos[0] = msg.x
    self.contact_pos[1] = msg.y
    self.contact_pos[2] = msg.z
    self.detect_contact(msg, countThresh=10)
    
  def patient_contact_norm_cb(self, msg: Vector3):
    # if msg.x != -1 and msg.y != -1 or msg.z != -1:
    self.contact_norm[0] = msg.x
    self.contact_norm[1] = msg.y
    self.contact_norm[2] = msg.z

  def detect_contact(self, collision_msg: Vector3, countThresh):
    if collision_msg.x != -1 and collision_msg.y != -1 and collision_msg.z != -1:
        isContactTmp = True
    else:
        isContactTmp = False
    
    if isContactTmp is False:
        self.nonContactCounter = self.nonContactCounter + 1 if self.nonContactCounter < countThresh else countThresh
    else:
        self.nonContactCounter = self.nonContactCounter - 1 if self.nonContactCounter > 0 else 0

    if self.nonContactCounter >= countThresh:
        self.isContact = False
    else:
        self.isContact = True

  def onSensorUpdate(self):
    while not rospy.is_shutdown():
        self.contact_state_pub.publish(data = self.isContact)
        self.rate.sleep()


if __name__ == '__main__':
    contact_detector = DetectContact()
    contact_detector.onSensorUpdate()