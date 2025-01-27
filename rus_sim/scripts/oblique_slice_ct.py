#! /usr/bin/env python3
# =========================================================
# file name:    oblique_slice_ct.py
# description:  sample 2D CT imaging window for US simulation
# author:       Xihan Ma
# date:         2024-01-30
# =========================================================

import os
from time import perf_counter

import cv2
import numpy as np
import SimpleITK as sitk

import torch
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
from sensor_msgs.msg  import Image
from tf import TransformBroadcaster
from tf.transformations import euler_matrix, quaternion_from_euler
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from franka_msgs.msg import FrankaState

class ObliqueSliceCT():
    """
    TODO: 
    - check ee to tip transformation
    - parameterize model positioning
    - adjust CT HU range with dynamic reconfigure
    - smooth displayed slice
    """
    def __init__(self) -> None:
        # ===== read CT volume =====
        self.ct_folder = os.path.join(os.path.dirname(__file__), '../../assets/')
        self.ct_volumes = {
            0 : 'CTLiver.nii'
        }
        self.ct_vol_ID = 0
        self.ct_volume, self.ct_vol_spacing, self.ct_vol_orig = \
            self.load_volume(os.path.join(self.ct_folder, self.ct_volumes[self.ct_vol_ID]))
        assert self.ct_volume.size > 0
        self.HU_LOWER = -1000       # air
        self.HU_UPPER = 800         # bone
        print(f'volume size: {self.ct_volume.shape}')
        print(f'volume data type: {self.ct_volume.dtype}')
        print(f'volume spacing: {self.ct_vol_spacing}')
        print(f'volume orgin: {self.ct_vol_orig}')
        print(f'CT background intensity: {self.HU_LOWER}')
        self.SLICE_HEIGHT = self.ct_volume[0,:,:].shape[0]
        self.SLICE_WIDTH = self.ct_volume[0,:,:].shape[1]
        self.slice = self.HU_LOWER*np.ones((self.SLICE_HEIGHT, self.SLICE_WIDTH), dtype=np.float32)

        # ===== get frame transformations =====
        self.contact_norm = np.zeros(3)
        self.contact_pos = np.zeros(3)
        # world to patient transform
        patient_orig = np.array([0.227, 0.0, 0.9])      # [m]
        T_w_pat = euler_matrix(-1.5707, 0.0, 1.5707).astype(np.float64)
        T_w_pat[:3, -1] = patient_orig
        self.T_pat_w = np.linalg.inv(T_w_pat)
        print(f'T_pat_w: \n{self.T_pat_w}')
        # self.patient_tf = TransformBroadcaster()
        # world to robot base transform
        arm_orig = np.array([0.0, 0.5, 0.79])  
        self.T_w_O = euler_matrix(0.0, 0.0, -1.5707).astype(np.float64)
        self.T_w_O[:3, -1] = arm_orig
        print(f'T_w_O: \n{self.T_w_O}')
        # robot base to ee transform
        self.T_O_ee = np.eye(4, dtype=np.float64)
        # robot ee to probe tip transform
        self.T_ee_tip = np.eye(4, dtype=np.float64)
        self.T_ee_tip[2, -1] = -0.2484/2 # prev:-0.2484/2, 0.2484. >0: closer to bottom; <0: closer to top

        # ===== initialize oblique slicing =====
        # pixel locations under end-effector frame [m]
        pc_x_ee, pc_z_ee = \
            np.meshgrid(np.linspace(-self.SLICE_WIDTH/2 * self.ct_vol_spacing[2], self.SLICE_WIDTH/2 * self.ct_vol_spacing[2], self.SLICE_WIDTH),
                        np.linspace(0, self.SLICE_HEIGHT * self.ct_vol_spacing[0], self.SLICE_HEIGHT),
                        indexing='ij')
        pc_x_ee = 1e-3*np.reshape(pc_x_ee, (1, pc_x_ee.size)).astype(np.float64)   # flatten & convert to meters
        pc_y_ee = np.zeros_like(pc_x_ee, dtype=np.float64)
        pc_z_ee = 1e-3*np.reshape(pc_z_ee, (1, pc_z_ee.size)).astype(np.float64)
        self.pc_tip = np.vstack((pc_x_ee, pc_y_ee, pc_z_ee, np.ones_like(pc_x_ee, dtype=np.float64)))
        # print(pc_x_ee)
        # print(pc_z_ee)
        # pixel locations under patient frame [mm]
        self.pc_x_pat = np.zeros(self.SLICE_HEIGHT*self.SLICE_WIDTH)
        self.pc_y_pat = np.zeros(self.SLICE_HEIGHT*self.SLICE_WIDTH)
        self.pc_z_pat = np.zeros(self.SLICE_HEIGHT*self.SLICE_WIDTH)
        self.calc_img_plane_coord()

        self.isContact = False

        # ===== initialize ROS node =====
        rospy.init_node('ct_oblique_slicing', anonymous=True)
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.franka_pose_cb)
        # rospy.Subscriber('/patient/contact/position', Vector3, self.patient_contact_pos_cb)
        # rospy.Subscriber('/patient/contact/normal', Vector3, self.patient_contact_norm_cb)
        rospy.Subscriber('/patient/contact/isContact', Bool, self.contact_state_cb)
        self.ct_slice_pub = rospy.Publisher('/patient/ct_slice', Image, queue_size=100)
        self.rate = rospy.Rate(30)
        self.set_volume_params()
        rospy.loginfo('finish initializing ROS node')

    def set_volume_params(self):
        try:
            if not rospy.is_shutdown():
                rospy.set_param('/CT/spacing', self.ct_vol_spacing)
                rospy.set_param('/CT/dim', self.ct_volume[0,:,:].shape)
                rospy.loginfo('set CT volume parameters')
        except Exception as e:
            print(e)

    def load_volume(self, volume_path):
        """
        unit: [mm]
        """
        volume = sitk.ReadImage(volume_path)
        return \
            sitk.GetArrayFromImage(volume), \
            volume.GetSpacing(), \
            volume.GetOrigin()

    def franka_pose_cb(self, msg: FrankaState):
        EE_pos = msg.O_T_EE  # inv 4x4 matrix
        self.T_O_ee = np.array([EE_pos[0:4], EE_pos[4:8], EE_pos[8:12], EE_pos[12:16]]).transpose()
    
    # def patient_contact_pos_cb(self, msg: Vector3):
    #     self.contact_pos[0] = msg.x
    #     self.contact_pos[1] = msg.y
    #     self.contact_pos[2] = msg.z
    
    # def patient_contact_norm_cb(self, msg: Vector3):
    #     self.contact_norm[0] = msg.x
    #     self.contact_norm[1] = msg.y
    #     self.contact_norm[2] = msg.z

    def contact_state_cb(self, msg: Bool):
        self.isContact = msg.data

    def calc_img_plane_coord(self):
        """
        Get imaging plane coordinates under patient frame.
        """
        # TODO: numpy matmul causes 3d haptics to fail, use torch instead
        
        # pc_pat = self.T_pat_w @ self.T_w_O @ self.T_O_ee @ self.T_ee_tip @ self.pc_tip

        T_pat_O = np.matmul(self.T_pat_w.copy(), self.T_w_O.copy())
        T_pat_ee = np.matmul(T_pat_O.copy(), self.T_O_ee.copy())
        T_pat_tip = np.matmul(T_pat_ee.copy(), self.T_ee_tip.copy())
        
        # pc_pat = np.matmul(T_pat_tip.copy(), self.pc_tip.copy())  

        A = torch.tensor(T_pat_tip.copy()).cuda()
        B = torch.tensor(self.pc_tip.copy()).cuda()
        pc_pat = torch.matmul(A, B).clone().cpu().numpy()

        # print(f'T_pat_ee: \n{self.T_pat_w @ self.T_w_O @ self.T_O_ee}')
        self.pc_x_pat = 1e3 * pc_pat[0, :]      # convert to mm
        self.pc_y_pat = 1e3 * pc_pat[1, :]
        self.pc_z_pat = 1e3 * pc_pat[2, :]

    def oblique_slice(self):
        """
        Perform oblique slicing.
        """
        # get voxel coordinates of the slicing plane
        px_x_pat = np.round((self.pc_x_pat + self.ct_vol_orig[0]) / self.ct_vol_spacing[0]).astype(int)
        px_y_pat = np.round((self.pc_y_pat + self.ct_vol_orig[1]) / self.ct_vol_spacing[1]).astype(int)
        px_z_pat = np.round((self.pc_z_pat + self.ct_vol_orig[2]) / self.ct_vol_spacing[2]).astype(int)

        # print(f'sample point range (patient frame) [mm]:')
        # print(f'x: {np.min(self.pc_x_pat)} --- {np.max(self.pc_x_pat)}')
        # print(f'y: {np.min(self.pc_y_pat)} --- {np.max(self.pc_y_pat)}')
        # print(f'z: {np.min(self.pc_z_pat)} --- {np.max(self.pc_z_pat)}')

        # print(f'voxel index range:')
        # print(f'x: {np.min(px_x_pat)} --- {np.max(px_x_pat)}')
        # print(f'y: {np.min(px_y_pat)} --- {np.max(px_y_pat)}')
        # print(f'z: {np.min(px_z_pat)} --- {np.max(px_z_pat)}')

        # find valid indices within the volume bound
        valid_cond = (px_x_pat >= 0) & (px_x_pat < self.ct_volume.shape[2]) & \
                     (px_y_pat >= 0) & (px_y_pat < self.ct_volume.shape[1]) & \
                     (px_z_pat >= 0) & (px_z_pat < self.ct_volume.shape[0])

        ind = np.nonzero(valid_cond)[0]
        rows, cols = np.unravel_index(ind, self.slice.shape)
        self.slice[rows, cols] = self.ct_volume[px_z_pat[ind], px_y_pat[ind], px_x_pat[ind]]

        self.slice = np.fliplr(self.slice[::-1, ::-1].T)

    def norm_ct_intensity(self):
        ct_tmp = self.slice.copy()
        ct_tmp = cv2.normalize(ct_tmp, None, 1, 0, cv2.NORM_MINMAX)
        self.slice = ct_tmp.copy()

    def update_ct_slice(self) -> None:
        """
        performs the following steps to update CT slice:
        1. calculate CT imaging plane coordinates
        2. slice the CT volume using the imaging plane
        3. normalize CT intensity
        3. (ongoing) smooth CT slice
        """
        self.calc_img_plane_coord()
        self.oblique_slice()
        self.norm_ct_intensity()

    def onSensorUpdate(self):

        # # ========== debug ==========
        # pointcloud_pub = rospy.Publisher('/img_plane_pcd', PointCloud2, queue_size=10)
        # pointcloud_msg = PointCloud2()
        # pointcloud_msg.header.stamp = rospy.Time.now()
        # pointcloud_msg.header.frame_id = 'patient'
        # # # ===========================
    
        while not rospy.is_shutdown():

            # R_w_p = quaternion_from_euler(-1.5707, 0.0, 1.5707)
            # R_w_p /= np.linalg.norm(R_w_p)
            # self.patient_tf.sendTransform((0.227, 0.0, 0.9),
            #                             R_w_p,
            #                             rospy.Time.now(),
            #                             "patient",
            #                             "world")
            
            self.slice = self.HU_LOWER*np.ones((self.SLICE_HEIGHT, self.SLICE_WIDTH), dtype=np.float32)    # reset frame
            if self.isContact:
                # rospy.loginfo('contact detected')
                self.update_ct_slice()    # debug: causes 3d haptics to fail
                
                # # ========== debug ==========
                # points = np.zeros((self.SLICE_HEIGHT*self.SLICE_WIDTH, 3))
                # for pnt in range(self.SLICE_HEIGHT*self.SLICE_WIDTH):
                #     points[pnt, 0] = 1e-3 * self.pc_x_pat[pnt].astype(np.float32)
                #     points[pnt, 1] = 1e-3 * self.pc_y_pat[pnt].astype(np.float32)
                #     points[pnt, 2] = 1e-3 * self.pc_z_pat[pnt].astype(np.float32)
                # points_list = points.tolist()
                # pointcloud_msg = pc2.create_cloud_xyz32(header=pointcloud_msg.header, points=points_list)
                # pointcloud_msg.header.stamp = rospy.Time.now()
                # pointcloud_pub.publish(pointcloud_msg)
                # # ===========================

            else:
                # rospy.loginfo('no contact detected')
                # self.slice = np.random.randint(0, 256, size=(self.SLICE_HEIGHT, self.SLICE_WIDTH)).astype(np.float32)
                self.slice = np.zeros((self.SLICE_HEIGHT, self.SLICE_WIDTH), dtype=np.float32)

                # # ========== debug ==========
                # points_list = [[0,0,0]] 
                # pointcloud_msg = pc2.create_cloud_xyz32(header=pointcloud_msg.header, points=points_list)
                # pointcloud_msg.header.stamp = rospy.Time.now()
                # pointcloud_pub.publish(pointcloud_msg)
                # # ===========================

            self.ct_slice_pub.publish(CvBridge().cv2_to_imgmsg(self.slice, encoding="32FC1"))
            self.rate.sleep()

if __name__ == '__main__':
    oblique_slicer = ObliqueSliceCT()
    oblique_slicer.onSensorUpdate()
