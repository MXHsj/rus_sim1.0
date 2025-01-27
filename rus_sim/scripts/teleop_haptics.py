#!/usr/bin/env python3
# =========================================================
# file name:    teleop_haptics.py
# description:  send teleop commands to rus_sim and receive force feedback
# author:       Xihan Ma, Nicholas Moy
# date:         2024-02-29
# =========================================================

from copy import copy

import numpy as np
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
import tf.transformations as tr

from franka_msgs.msg import FrankaState
from geomagic_control.msg import DeviceFeedback


class StylusWorkspace():
    """ define workspace of the stylus
    """
    POS_X_LOWER = -100
    POS_X_UPPER = 100
    POS_Y_LOWER = 0
    POS_Y_UPPER = 100
    POS_Z_LOWER = -25
    POS_Z_UPPER = 80

    def is_in_workspace(self, curr_pos: PoseStamped):
        is_in_pos_x = (curr_pos.pose.position.x > self.POS_X_LOWER and curr_pos.pose.position.x < self.POS_X_UPPER)
        is_in_pos_y = (curr_pos.pose.position.y > self.POS_Y_LOWER and curr_pos.pose.position.x < self.POS_Y_UPPER)
        is_in_pos_z = (curr_pos.pose.position.z > self.POS_Z_LOWER and curr_pos.pose.position.z < self.POS_Z_UPPER)

        is_in_ = is_in_pos_x and is_in_pos_y and is_in_pos_z
        return is_in_
        

class TeleopHaptics():
    """
    TODO:
    1. init stylus pose
    2. 
    """
    def __init__(self, en_feedback = True) -> None:
        #for correcting limits to ensure we don't go into singularity or the patient or the table and stay in a reasonable workspace
        self.limit_correction = True

        #overides skipping of position updates
        self.Overide_input_skips = False
        #scaling factor for packets being sent to gazbo (does not like 1000hz)
        self.sendUpdateEveryXticks = 25
        self.tick = 0

        #motion scalling
        self.scalling_factorX = 1/190
        self.scalling_factorY = 1/200
        self.scalling_factorZ = 1/250

        #motion ofsets
        self.Xoffset = 0.5
        self.Zoffset = -0.3
        #tip of ultrasound to ee offset matrix
        self.tip_matrix = tr.identity_matrix()
        self.tip_matrix[2][3] = 0.25

        #orientation rotation matrix
        self.M0 = tr.euler_matrix(1.57 + 3.14,1.57,3.14,'rxyz')


        #feedback offsets
        self.fzOffset = 0

        self.robotPoseUncorrected = PoseStamped()
        
        self.T_O_ee = tr.identity_matrix()      # T from robot base to ee
        self.T_O_ee_d = PoseStamped()           # desired T from robot base to ee
        self.T_b_sty = PoseStamped()            # T from geomagic based to stylus

        self.EN_FORCE_FB = en_feedback
        self.force_deadband = [0.01, 0.01, 0.01]    # [0.03, 0.03, 0.02]
        self.force_fb_gains = [10, 10, 10]          # [20, 20, 20]
        self.force_fb = DeviceFeedback()
        self.force_fb_last = DeviceFeedback()
        self.force_fb_smooth_factor = 0.8
        self.pos_err_queue_sz = 20
        self.pos_err_queue = [[0, 0, 0] for i in range(self.pos_err_queue_sz)]

        self.stylus_ws = StylusWorkspace()
        self.isUpdate = False

        # ========== init ROS node ==========
        rospy.init_node('teleop_haptic', anonymous=True)
        rospy.Subscriber("/Geomagic/pose", PoseStamped, self.geomagic_pose_cb)
        rospy.Subscriber('franka_state_controller/franka_states', FrankaState, self.franka_pose_cb)

        self.equi_pose_pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=1)
        self.force_fb_pub = rospy.Publisher('/Geomagic/force_feedback', DeviceFeedback, queue_size=1)
        self.teleop_flag_pub = rospy.Publisher('/isUpdatePoseTeleop', Bool, queue_size=1)
        self.rate = rospy.Rate(20)
        rospy.loginfo(f"system ready, feedback = {self.EN_FORCE_FB}")
        # ===================================

    def franka_pose_cb(self, msg: FrankaState):
        EE_pos = msg.O_T_EE  # inv 4x4 matrix
        self.T_O_ee = np.array([EE_pos[0:4], EE_pos[4:8], EE_pos[8:12], EE_pos[12:16]]).transpose()
        

    def geomagic_pose_cb(self, msg: PoseStamped):
        """ receive geomagic stylus pose & convert pose to w.r.t robot base
        """
        self.T_b_sty = copy(msg)
        self.isUpdate = self.stylus_ws.is_in_workspace(self.T_b_sty)

    def update_teleop_flag(self):
        if self.isUpdate:
            self.teleop_flag_pub.publish(data=True)
        else:
            self.teleop_flag_pub.publish(data=False)

    def calc_desired_pose(self):
        """ caculate desired robot ee pose from geomagic stylus pose
        """
        self.T_b_sty.header.frame_id = 'panda_link0'

        M = tr.quaternion_matrix([self.T_b_sty.pose.orientation.x, self.T_b_sty.pose.orientation.y, self.T_b_sty.pose.orientation.z, self.T_b_sty.pose.orientation.w])
        M1 = tr.numpy.matmul(self.M0, M)

        #Invert the Z axis
        E = tr.euler_from_matrix(M1, 'sxyz')
        Q3 = tr.quaternion_from_euler(-E[0], E[1] + 3.14, E[2], 'sxyz')

        #calculate tip offset
        tip_offset = tr.numpy.matmul(M, self.tip_matrix)
        #scale and offset the motion to be aproprate to the robot
        self.T_O_ee_d.pose.position.x = self.T_b_sty.pose.position.z * self.scalling_factorZ + self.Xoffset + tip_offset[2][3]        
        self.T_O_ee_d.pose.position.y = self.T_b_sty.pose.position.x * self.scalling_factorY + tip_offset[0][3]        
        self.T_O_ee_d.pose.position.z = self.T_b_sty.pose.position.y * self.scalling_factorZ + self.Zoffset + tip_offset[1][3]

        # #append the orientation to the message
        self.T_O_ee_d.pose.orientation.x = Q3[0]
        self.T_O_ee_d.pose.orientation.y = Q3[1]
        self.T_O_ee_d.pose.orientation.z = Q3[2]
        self.T_O_ee_d.pose.orientation.w = Q3[3]
        # print(f'desired pose: {self.T_O_ee_d}')

        self.equi_pose_pub.publish(self.T_O_ee_d)

    def update_force_feedback(self):
        """
        ongoing
        """   
        self.force_fb.position.x = 0
        self.force_fb.position.y = 0
        self.force_fb.position.z = 0
        self.force_fb.lock = [False]

        pos_error = [self.T_O_ee[0, -1] - self.T_O_ee_d.pose.position.x, \
                     self.T_O_ee[1, -1] - self.T_O_ee_d.pose.position.y, \
                     self.T_O_ee[2, -1] - self.T_O_ee_d.pose.position.z]
        
        self.pos_err_queue.append(pos_error)
        if len(self.pos_err_queue) > self.pos_err_queue_sz:
            self.pos_err_queue.pop(0)
        
        pos_x_err = np.mean(np.array(self.pos_err_queue)[:, 0])
        pos_y_err = np.mean(np.array(self.pos_err_queue)[:, 1])
        pos_z_err = np.mean(np.array(self.pos_err_queue)[:, 2])
        
        # for i in range(3):
        #     pos_error[i] = 0.0 if abs(pos_error[i] < self.force_deadband[i]) else pos_error[i]

        print(f'pos err\n x:{pos_error[0]:.3f} y:{pos_error[1]:.3f} z:{pos_error[2]:.3f}')
        print(f'pos err\n x:{pos_x_err:.3f} y:{pos_y_err:.3f} z:{pos_z_err:.3f}')

        self.force_fb.force.x = self.force_fb_smooth_factor * self.force_fb_gains[1] * pos_error[1] + \
                                (1-self.force_fb_smooth_factor) * self.force_fb_last.force.x
        self.force_fb.force.y = self.force_fb_smooth_factor * self.force_fb_gains[2] * pos_error[2] + \
                                (1-self.force_fb_smooth_factor) * self.force_fb_last.force.y
        self.force_fb.force.z = self.force_fb_smooth_factor * self.force_fb_gains[0] * pos_error[0] + \
                                (1-self.force_fb_smooth_factor) * self.force_fb_last.force.z
        self.force_fb_last = copy(self.force_fb)

        # # ========== debug =========
        # wall_distance = self.T_O_ee_d.pose.position.z - 0.0
        # wall_distance = self.T_O_ee[2, -1] - 0.0
        # if abs(wall_distance) < 0.35:
        #     self.force_fb.force.y = 1.0
        # else:
        #     self.force_fb.force.y = 0.0
        # print(f'wall distance: {wall_distance:.4f}, repulsive force: {self.force_fb.force.y}')
        # # ==========================

        # print(self.force_fb)
        self.force_fb_pub.publish(self.force_fb)
        
    def onUpdate(self):
        """
        """
        self.calc_desired_pose()

        while not rospy.is_shutdown():
            self.update_teleop_flag()

            if self.isUpdate:
                self.calc_desired_pose()
                if self.EN_FORCE_FB:
                    self.update_force_feedback()

            self.rate.sleep()


if __name__ == '__main__':
    teleop = TeleopHaptics(en_feedback=False)
    teleop.onUpdate()