#! /usr/bin/env python3
# =========================================================
# file name:    asee1_panda_bridge.py
# description:  streaming sensor data from laser scan & publish to asee1 topics
# author:       Xihan Ma
# date:         2024-01-24
# =========================================================

import numpy as np

import rospy
from tf import TransformBroadcaster
from tf.transformations import euler_from_matrix, quaternion_from_euler, quaternion_from_matrix
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan

from franka_msgs.msg import FrankaState


class ASEE1PandaBridge():

    def __init__(self) -> None:

        # =============== params ===============
        self.ringR = 58             # sensor ring radius [mm]
        self.ringL = 85             # sensor ring height [mm]
        self.ringX = [self.ringR, -self.ringR, 0, 0]
        self.ringY = [0, 0, self.ringR, -self.ringR]
        self.NUM_SENSORS = 4
        self.BUFFER_SIZE = 10
        self.MAX_DIST = 200
        # ======================================
        
        self.T_O_ee = np.eye(4)

        self.dist_raw = np.zeros((4), dtype=np.float32) #[float('nan'), float('nan'), float('nan'), float('nan')]
        self.dist_filtered = [0, 0, 0, 0]
        self.dist_buffer = [[float('nan')]*self.BUFFER_SIZE, [float('nan')]*self.BUFFER_SIZE,
                            [float('nan')]*self.BUFFER_SIZE, [float('nan')]*self.BUFFER_SIZE]
        self.buf_count = 0

        self.norm = [0, 0, 0]

        self.ee_axis_name = "panda_asee1"
        self.contact_axis_name = "contact_plane"

        rospy.init_node('asee1_panda_interface', anonymous=True)
        rospy.Subscriber('/asee1/sensor1/laser', LaserScan, queue_size=1, callback=self.s1_laser_cb)
        rospy.Subscriber('/asee1/sensor2/laser', LaserScan, queue_size=1, callback=self.s2_laser_cb)
        rospy.Subscriber('/asee1/sensor3/laser', LaserScan, queue_size=1, callback=self.s3_laser_cb)
        rospy.Subscriber('/asee1/sensor4/laser', LaserScan, queue_size=1, callback=self.s4_laser_cb)

        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.franka_pose_cb)
        

        self.VL53L0X_dist_pub = rospy.Publisher('VL53L0X/distance', Float32MultiArray, queue_size=1)
        self.VL53L0X_norm_pub = rospy.Publisher('VL53L0X/normal', Float32MultiArray, queue_size=1)
        self.contact_pnt_tf = TransformBroadcaster()
        self.rate = rospy.Rate(30)

    def regulate_sensor_reading(self, measure):
        if measure > self.MAX_DIST or np.isnan(measure) or np.isinf(measure):
            measure = self.MAX_DIST
        return measure

    def reject_outliers(self, data, m=2):
        return data[abs(data-np.nanmedian(data)) < m*np.nanstd(data)]

    def estimate_normal(self):
        # =============== method1: ===============
        P = np.ones((self.NUM_SENSORS, 3))
        # norms = np.zeros((self.NUM_SENSORS, 3))
        P[:, 0] = self.ringX
        P[:, 1] = self.ringY
        # P[:, 2] = -self.ringL + np.array(self.dist_filtered)
        P[:, 2] = self.MAX_DIST - np.array(self.dist_filtered)
        centroid = np.mean(P, axis=0)
        # print(P)
        norm1 = np.cross(P[0, :]-centroid, P[2, :]-centroid)
        norm2 = np.cross(P[1, :]-centroid, P[3, :]-centroid)
        norm3 = np.cross(P[1, :]-centroid, P[2, :]-centroid)
        norm4 = np.cross(P[3, :]-centroid, P[0, :]-centroid)
        norms = norm1 + norm2 + norm3 + norm4
        # print(norms)
        # print(norm1 + norm2 + norm3)
        # for i in range(self.NUM_SENSORS):
        #     norms[i, :] = np.cross(P[i, :], P[i+1, :]) if i < self.NUM_SENSORS - 1 else np.cross(P[i, :], P[0, :])
        #     if np.isnan(norms[i, :]).any():
        #         norms[i, :] = [0, 0, 0]
        # self.norm = np.sum(norms, axis=0)/np.linalg.norm(np.sum(norms, axis=0))
        self.norm = norms/np.linalg.norm(norms)
        # ========================================

        # # =============== method2: ===============
        # # A = np.ones((self.NUM_SENSORS, self.NUM_SENSORS))
        # A = np.zeros((4, 3), dtype=np.float32)
        # A[:, 0] = self.ringX
        # A[:, 1] = self.ringY
        # # A[:, 2] = -self.ringL + np.array(self.dist_filtered)
        # A[:, 2] = self.MAX_DIST - np.array(self.dist_filtered)
        # A = np.hstack((A, np.ones((4, 1), dtype=np.float32)))
        # # print(A)
        # _, _, V = np.linalg.svd(A)
        # # print(V)
        # x = V[:3, 3]
        # # if x[2] < 0:
        # #     x = -x
        # self.norm = x/np.linalg.norm(x)
        # # ========================================

    def s1_laser_cb(self, msg):
        self.dist_raw[0] = self.regulate_sensor_reading(msg.ranges[0]*1e3)

    def s2_laser_cb(self, msg):
        self.dist_raw[1] = self.regulate_sensor_reading(msg.ranges[0]*1e3)

    def s3_laser_cb(self, msg):
        self.dist_raw[2] = self.regulate_sensor_reading(msg.ranges[0]*1e3)

    def s4_laser_cb(self, msg):
        self.dist_raw[3] = self.regulate_sensor_reading(msg.ranges[0]*1e3)

    def franka_pose_cb(self, msg):
        EE_pos = msg.O_T_EE  # inv 4x4 matrix
        self.T_O_ee = np.array([EE_pos[0:4], EE_pos[4:8], EE_pos[8:12], EE_pos[12:16]]).transpose()

    def boardcast_contact_pnt(self):
        self.estimate_normal()
        R_O_ee = self.T_O_ee[:3, :3]
        T_O_contact = self.T_O_ee.copy()
        T_O_contact[2, :3] = self.norm
        T_O_contact[1, :3] = np.cross(T_O_contact[0, :3], T_O_contact[2, :3])
        rpy = euler_from_matrix(T_O_contact)
        self.contact_pnt_tf.sendTransform((0, 0, np.nanmean(self.dist_filtered)*1e-3),
                                        quaternion_from_euler(rpy[0], rpy[1], rpy[2]),
                                        rospy.Time.now(),
                                        self.contact_axis_name,
                                        self.ee_axis_name)

    def onSensorUpdate(self):
        while not rospy.is_shutdown():
            
            # dist_filtered = [0, 0, 0, 0]
            # for i in range(self.NUM_SENSORS):
            #     self.dist_buffer[i][self.buf_count] = self.dist_raw[i]
            #     filtered = self.reject_outliers(np.array(self.dist_buffer[i]), m=3.0)
            #     dist_filtered[i] = np.nanmean(filtered)

            # self.dist_filtered = self.comp_sensor_err(dist_filtered)
            self.dist_filtered = self.dist_raw.copy()

            print(f'sensor1:{self.dist_filtered[0]: .1f}[mm], sensor2:{self.dist_filtered[1]: .1f}[mm]',
                  f'sensor3:{self.dist_filtered[2]: .1f}[mm], sensor4:{self.dist_filtered[3]: .1f}[mm]')
            # print(f'normal vector: {self.norm}')
            self.VL53L0X_dist_pub.publish(Float32MultiArray(data=self.dist_filtered))
            self.VL53L0X_norm_pub.publish(Float32MultiArray(data=self.norm))

            self.boardcast_contact_pnt()

            self.buf_count = self.buf_count + 1 if self.buf_count < self.BUFFER_SIZE-1 else 0
            self.rate.sleep()

if __name__ == '__main__':
    asee1_bridge = ASEE1PandaBridge()
    asee1_bridge.onSensorUpdate()