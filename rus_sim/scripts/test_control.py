#!/usr/bin/env python3

# ========== this is a testing script ==========
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from tf.transformations import quaternion_from_matrix

from franka_msgs.msg import FrankaState
from matplotlib import pyplot as plt

class TestControl():

    def __init__(self) -> None:
        rospy.init_node('pos_control_test', anonymous=True)
        rospy.Subscriber('franka_state_controller/franka_states', FrankaState, self.franka_pose_cb)
        rospy.Subscriber('franka_state_controller/F_ext', WrenchStamped, self.F_ext_cb)
        self.equi_pose_pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=1)
        self.rate = rospy.Rate(10)

        self.T_O_ee = np.eye(4)
        self.T_O_EE_d = np.array([[0.0, -1.0, 0.0, 0.5],
                                [-1.0, 0.0, 0.0, -0.1],
                                [0.0, 0.0, -1.0, 0.45],
                                [0.0, 0.0, 0.0, 1.0]])
        self.T_home = np.array([[0.0, -1.0, 0.0, 0.4],
                                [-1.0, 0.0, 0.0, 0],
                                [0.0, 0.0, -1.0, 0.6],
                                [0.0, 0.0, 0.0, 1.0]])
        self.Fz = 0
        self.Fz_d = 5
        self.Fz_queue = []
    
    def franka_pose_cb(self, msg):
        EE_pos = msg.O_T_EE  # inv 4x4 matrix
        self.T_O_ee = np.array([EE_pos[0:4], EE_pos[4:8], EE_pos[8:12], EE_pos[12:16]]).transpose()

    def F_ext_cb(self, msg):
        self.Fz = abs(msg.wrench.force.z)
        if len(self.Fz_queue) < 100:
            self.Fz_queue.append(self.Fz)
        else:
            self.Fz_queue.pop(0)
            self.Fz_queue.append(self.Fz)
        # print(self.Fz_queue)

    def contact_control(self):
        Kp = -0.032
        F_err = self.Fz_d - abs(self.Fz)
        z_new = self.T_O_ee[2, -1] + Kp*F_err
        self.T_O_EE_d[2, -1] = z_new
        print(f'contact force: {abs(self.Fz)}, dz: {Kp*F_err}')

    def move2pose(self, pose):
        res_err = 0.008 
        quat = quaternion_from_matrix(pose)
        pose_goal = PoseStamped()
        pose_goal.header.stamp = rospy.Time.now()
        pose_goal.pose.orientation.x = quat[0]
        pose_goal.pose.orientation.y = quat[1]
        pose_goal.pose.orientation.z = quat[2]
        pose_goal.pose.orientation.w = quat[3]
        pose_goal.pose.position.x = pose[0, -1]
        pose_goal.pose.position.y = pose[1, -1]
        pose_goal.pose.position.z = pose[2, -1]
        # return pose_goal
        while not rospy.is_shutdown():
            self.equi_pose_pub.publish(pose_goal)
            pos_err_x = abs(self.T_O_ee[0, -1] - pose[0, -1])
            pos_err_y = abs(self.T_O_ee[1, -1] - pose[1, -1])
            pos_err_z = abs(self.T_O_ee[2, -1] - pose[2, -1])
            # print(f'x err: {pos_err_x}, y err: {pos_err_y}, z err: {pos_err_z}')
            if pos_err_x < res_err and pos_err_y < res_err and pos_err_z < res_err:
                print(f'goal pose reached: \n{pose}')
                break

        self.equi_pose_pub.publish(pose_goal)

    def onUpdate(self):
        entry_pose = self.T_O_EE_d.copy()
        entry_pose[1, -1] -= 0.01
        entry_pose[2, -1] -= 0.03

        start_pose = entry_pose.copy()
        start_pose[2, -1] -= 0.0700 # 0.070086
        
        end_pose1 = start_pose.copy()
        end_pose1[1, -1] -= 0.06

        end_pose2 = start_pose.copy()
        end_pose2[1, -1] += 0.04

        # ========== 
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 20)
        line, = ax.plot(rospy.Time.now().to_sec(), 0)
        plt.grid(True)
        # line.set_data(np.linspace(0, len(self.Fz_queue), len(self.Fz_queue)), np.array(self.Fz_queue))
        # plt.draw()
        # plt.pause(0.0001)

        self.move2pose(pose=self.T_home)
        rospy.sleep(1)
        self.move2pose(pose=entry_pose)
        rospy.sleep(2)
        self.move2pose(pose=start_pose)
        rospy.sleep(1)
        self.move2pose(pose=end_pose1)
        rospy.sleep(1)
        self.move2pose(pose=end_pose2)

        # while not rospy.is_shutdown():
            
            # ==========
            # self.move2pose(pose=end_pose)
            # self.rate.sleep()

if __name__ == '__main__':
    test_instance = TestControl()
    test_instance.onUpdate()