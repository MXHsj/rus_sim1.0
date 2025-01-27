#!/usr/bin/env python3
# =========================================================
# file name:    verify_normal_estimate.py
# description:  verify normal estimation accuracy 
# author:       Xihan Ma
# date:         2024-04-01
# =========================================================
import os

from stl import mesh
import numpy as np
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion, euler_matrix
import rospy

def calc_vec_diff(vec1: np.ndarray, vec2: np.ndarray) -> np.float64:
  '''
  calculate angular difference between vectors
  '''
  proj = np.dot(vec1, vec2)
  mag_vec1 = np.linalg.norm(vec1)
  mag_vec2 = np.linalg.norm(vec2)
  cos_ang = proj / (mag_vec1 * mag_vec2)
  angle = np.arccos(np.clip(cos_ang, -1.0, 1.0), dtype=np.float64)
  return np.degrees(angle)

class VerifyNormalEstimate():
  '''
  1. estimate normal vector
  2. check contact status
  3. get contact point & normal (ground truth)
  4. calculate normal estimation error
  5. export data
  '''

  def __init__(self) -> None:
    '''
    '''
    patient_mesh_file = '../models/patient_CT/model.stl'
    self.mesh_data = mesh.Mesh.from_file(patient_mesh_file)

    # TODO: access tf from gazebo link states
    patient_orig = np.array([0.227, 0.0, 0.9])      # [m]
    self.T_w_pat = euler_matrix(-1.5707, 0.0, 1.5707).astype(np.float64)
    self.T_w_pat[:3, -1] = patient_orig
    self.T_pat_w = np.linalg.inv(self.T_w_pat)
    print(f'T_pat_w: \n{self.T_pat_w}')

    arm_orig = np.array([0.0, 0.5, 0.79])  
    self.T_w_O = euler_matrix(0.0, 0.0, -1.5707).astype(np.float64)
    self.T_w_O[:3, -1] = arm_orig
    print(f'T_w_O: \n{self.T_w_O}')

    self.contact_pos_gt = np.zeros(3)     # ground truth contact point
    self.contact_norm_gt = np.zeros(3)    # ground truth contact point normal
    self.contact_norm = np.zeros(3)       # estimated contact point normal
    self.isContact = False

    rospy.init_node('verify_normal_estimate', anonymous=True)
    rospy.Subscriber('/patient/contact/position', Vector3, self.patient_contact_pos_cb)
    rospy.Subscriber('/patient/contact/normal', Vector3, self.patient_contact_norm_cb)
    rospy.Subscriber('/patient/contact/isContact', Bool, self.patient_contact_status_cb)
    rospy.Subscriber('VL53L0X/normal', Float32MultiArray, self.asee_norm_cb)
    self.rate = rospy.Rate(1)

  def patient_contact_pos_cb(self, msg: Vector3):
    if msg.x != -1 and msg.y != -1 and msg.z != -1:
      self.contact_pos_gt[0] = msg.x
      self.contact_pos_gt[1] = msg.y
      self.contact_pos_gt[2] = msg.z
    
  def patient_contact_norm_cb(self, msg: Vector3):
    # if msg.x != -1 and msg.y != -1 and msg.z != -1:
    #   self.contact_norm_gt[0] = msg.x
    #   self.contact_norm_gt[1] = msg.y
    #   self.contact_norm_gt[2] = msg.z
    ...

  def patient_contact_status_cb(self, msg: Bool):
    self.isContact = msg.data

  def asee_norm_cb(self, msg: Float32MultiArray):
    self.contact_norm[0] = msg.data[0]
    self.contact_norm[1] = msg.data[1]
    self.contact_norm[2] = msg.data[2]

  def cvt_point_to_mesh(self):
    '''
    calculate point in world coord. to patient (mesh) coord.
    '''
    P_w = np.append(self.contact_pos_gt, 1.0)
    # print(f'contact point in wolrd: {P_w}')
    P_pat = self.T_pat_w @ P_w.T
    print(f'contact point on mesh: {P_pat[:3]}')
    return P_pat[:3]

  def cvt_norm_to_world(self, mesh_norm):
    N_w = (self.T_w_pat @ np.append(mesh_norm, 1))[:3]
    self.contact_norm_gt = N_w / np.linalg.norm(N_w)

  def estimate_mesh_normal(self):
    '''
    calculate normal vector of the closest mesh triangle to a point in the STL model
    '''
    # define point in mesh
    point = self.cvt_point_to_mesh()

    # find closest triangle
    # distances = np.linalg.norm(self.mesh_data.vectors.mean(axis=1) - point, axis=1)
    distances = np.linalg.norm(self.mesh_data.centroids - point, axis=1)

    closest_triangle_index = np.argmin(distances)
    closest_triangle = self.mesh_data.vectors[closest_triangle_index, :, :]
    # dist_min = np.linalg.norm(closest_triangle.mean() - point)
    print(f'closest triangle: \n{closest_triangle}')
    # print(f'closest distance: {distances.min()}')

    # compute normal vector of the triangle
    edge1 = closest_triangle[1] - closest_triangle[0]
    edge2 = closest_triangle[2] - closest_triangle[0]
    mesh_norm = np.cross(edge1, edge2)
    mesh_norm /= np.linalg.norm(mesh_norm)

    # mesh_norm = self.mesh_data.normals[closest_triangle_index]
    # mesh_norm /= np.linalg.norm(mesh_norm)

    print(f'normal vector on mesh: {mesh_norm}')
    self.cvt_norm_to_world(mesh_norm)

  def save_rec_data(self):
    ...

  def onUpdate(self):
    while not rospy.is_shutdown():
      if self.isContact:
        self.estimate_mesh_normal()
        angle = calc_vec_diff(self.contact_norm, self.contact_norm_gt)
        print(f'ground truth vec: {self.contact_norm_gt}\nestimated vec: {self.contact_norm}')
        # print(f'normal estimation error: {angle:.4f} deg')

      self.rate.sleep()
  

if __name__ == '__main__':
  pipeline = VerifyNormalEstimate()
  pipeline.onUpdate()
