#! /usr/bin/env python3
# =========================================================
# file name:    synthesize_us.py
# description:  simulate us from 2D CT slice
# author:       Xihan Ma
# date:         2024-02-07
# =========================================================
import os
import sys

import numpy as np
import cv2
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../../ct2us/im2im/'))

from unet.model import UNet_dilation, UNet, AttentionUNet
from utils.vis import array2tensor, tensor2array
from utils.predict import predict

model_weights = {0: 'ct2us_unet_dilation_cur.pth', 
                 1: 'ct2us_unet_cur.pth',
                 2: 'ct2us_attention_unet_cur.pth'
                }
model_names = {0: UNet_dilation,
               1: UNet,
               2: AttentionUNet
              }

def fill_hole(im_in: np.ndarray):
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill = im_in.copy()
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    res = (im_in | im_floodfill_inv)/255
    return res

class SimulateUS():
    """
    TODO:
    1. finalize US image size
    """
    def __init__(self, is_infer_us = None) -> None:
        self.CT_HEIGHT = None        
        self.CT_WIDTH = None
        self.CT_RES_AXI = None
        self.CT_RES_LAT = None
        # self.US_HEIGHT = 480        # default values
        # self.US_WIDTH = 640
        self.US_HEIGHT = 215        # default values
        self.US_WIDTH = 286
        self.US_FOV_AXI = 150       # axial field of view [mm]
        self.US_FOV_LAT = 200       # lateral field of view [mm]
        self.us_img_win = None      # bbox
        self.us_img = np.zeros((self.US_HEIGHT, self.US_WIDTH), dtype=np.float32)
        self.VIRTUE_CENTER = [320, -96]
        self.BEAM_ANGLE = [-33, 33]
        if is_infer_us is None:
            self.IS_INFER_US = True if torch.cuda.is_available() else False
        else:
            self.IS_INFER_US = is_infer_us
        self.modelID = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.join(os.path.dirname(__file__), '../../ct2us/im2im/model/'+model_weights[self.modelID])
        self.model = None

        self.isContact = False
        self.isValidCT = False
        
        # ===== initialize ROS node =====
        rospy.init_node('simulate_us', anonymous=True)
        rospy.Subscriber('/patient/ct_slice', Image, callback=self.ct_slice_cb)
        rospy.Subscriber('/patient/contact/isContact', Bool, self.contact_state_cb)
        self.us_sim_pub = rospy.Publisher('/patient/us_sim', Image, queue_size=100)
        self.rate = rospy.Rate(30)
        rospy.loginfo('finish initializing ROS node')

        # ===== initialize US image =====
        self.get_volume_info()
        self.ct_slice = np.zeros((self.CT_HEIGHT, self.CT_WIDTH), dtype=np.float32)
        self.CURVI_MSK = self.gen_curvi_msk()
        # cv2.imshow('curvi mask', self.CURVI_MSK)
        # cv2.waitKey(0)
        self.init_us_img()
        rospy.loginfo('finish initializing US image')

        # ===== load us sim model =====
        if self.IS_INFER_US:
            self.model = self.load_us_sim_model()
            rospy.loginfo('finish loading model from: %s', self.model_path)

    def get_volume_info(self):
        try:
            if not rospy.is_shutdown():
                spacing = rospy.get_param('/CT/spacing')
                dim = rospy.get_param('/CT/dim')
                self.CT_RES_AXI = spacing[0]
                self.CT_RES_LAT = spacing[1]
                self.CT_HEIGHT = dim[0]
                self.CT_WIDTH = dim[1]
                rospy.loginfo('CT volume info retrieved')
        except Exception as e:
            print(f'exception occurred whiling loading ct volume info: {e}')
            exit()

    def load_us_sim_model(self) -> torch.nn.Module:
        network = model_names[self.modelID]
        us_sim_net = network(n_channels=1, n_classes=1, bilinear=False)
        us_sim_net.to(device=self.device)
        us_sim_net.load_state_dict(torch.load(self.model_path))
        return us_sim_net

    def ct_slice_cb(self, msg: Image):
        if self.CT_HEIGHT is None or self.CT_WIDTH is None:
            self.CT_HEIGHT = msg.height
            self.CT_WIDTH = msg.width
        self.ct_slice = CvBridge().imgmsg_to_cv2(img_msg=msg, desired_encoding=msg.encoding)
        self.isValidCT = True if self.ct_slice.sum() != 0 else False

    def contact_state_cb(self, msg: Bool):
        self.isContact = msg.data

    def gen_curvi_msk(self) -> np.ndarray:
        """
        This is a temporary solution
        """
        # msk_file = '../../assets/curvi_msk_C3HD.png'
        msk_file = '../../assets/curvi_msk_P42v.png'
        msk_path = os.path.join(os.path.dirname(__file__), msk_file)
        if os.path.isfile(msk_path):
            curvi_msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            curvi_msk = np.multiply(1/curvi_msk.max(), curvi_msk).astype(np.float32)
        else:
            rospy.logwarn('US imaging window mask not exit, creating one ...')
            curvi_msk = np.zeros((self.US_HEIGHT, self.US_WIDTH), dtype=np.uint8)
            for th in np.arange(self.BEAM_ANGLE[0], self.BEAM_ANGLE[1], 0.1):
                for d in np.arange(120, 550, 0.15):
                    col = round(d*np.sin(np.deg2rad(th))) + self.VIRTUE_CENTER[0]
                    row = round(d*np.cos(np.deg2rad(th))) + self.VIRTUE_CENTER[1]
                    curvi_msk[row, col] = 1
            curvi_msk = fill_hole(curvi_msk)
            cv2.imwrite(msk_path, curvi_msk)
            rospy.loginfo('saved US imaging window mask to %s', msk_path)

        return curvi_msk

    def init_us_img(self):
        """
        TODO: account for distance from image boundary to tissue surface in CT?
        """
        axi = self.US_FOV_AXI * (1/self.CT_RES_AXI)
        lat = self.US_FOV_LAT * (1/self.CT_RES_LAT)
        fov_axi_st = 0
        fov_axi_ed = round(axi) if axi < self.CT_HEIGHT else self.CT_HEIGHT
        fov_lat_st = round(self.CT_WIDTH/2) - round(lat/2) if lat < self.CT_WIDTH else 0
        fov_lat_ed = round(self.CT_WIDTH/2) + round(lat/2) if lat < self.CT_WIDTH else self.CT_WIDTH - 1
        print(f'axi st: {fov_axi_st}, axi ed: {fov_axi_ed}, lat st: {fov_lat_st}, lat ed: {fov_lat_ed}')
        self.us_img_win = (fov_axi_st, fov_axi_ed, fov_lat_st, fov_lat_ed)
        us_img_raw = self.ct_slice[fov_axi_st:fov_axi_ed, fov_lat_st:fov_lat_ed]
        us_img_raw = cv2.normalize(us_img_raw, None, 1, 0, cv2.NORM_MINMAX)
        if self.IS_INFER_US:
            self.us_img = np.multiply(self.CURVI_MSK, cv2.resize(us_img_raw, (self.US_WIDTH, self.US_HEIGHT)))
        else:
            self.us_img = np.multiply(self.CURVI_MSK, cv2.resize(us_img_raw, (self.US_WIDTH, self.US_HEIGHT)))

    def update_us_img(self):
        """
        TODO: 
        1. account for distance from image boundary to tissue surface in CT?
        2. fix delayed US update when first pressed on the body
        """
        us_img_raw = np.random.rand(self.US_HEIGHT, self.US_WIDTH).astype(np.float32)
        # print(f'contact state: {self.isContact}, CT ready: {self.isValidCT}')

        if self.isContact and self.isValidCT:
            us_img_raw = self.ct_slice[self.us_img_win[0]:self.us_img_win[1], \
                                        self.us_img_win[2]:self.us_img_win[3]]
            us_img_raw = cv2.normalize(us_img_raw, None, 1, 0, cv2.NORM_MINMAX)

            if self.IS_INFER_US:
                us_img_tmp = np.multiply(self.CURVI_MSK, cv2.resize(us_img_raw, (self.US_WIDTH, self.US_HEIGHT)))
                us_img_tmp = array2tensor(us_img_tmp, device=self.device)
                self.us_img = tensor2array(self.model(us_img_tmp))
            else:
                self.us_img = np.multiply(self.CURVI_MSK, cv2.resize(us_img_raw, (self.US_WIDTH, self.US_HEIGHT)))
        
        else:
            self.us_img = np.multiply(self.CURVI_MSK, us_img_raw)

        return us_img_raw   # debug

    def onUpdate(self):
        while not rospy.is_shutdown():
            res = self.update_us_img()
            # ===== debug =====
            # cv2.imshow('us sim', self.us_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # =================

            self.us_sim_pub.publish(CvBridge().cv2_to_imgmsg(self.us_img, encoding="32FC1"))
            self.rate.sleep()
        

if __name__ == '__main__':
    us_synth = SimulateUS(is_infer_us=True)
    us_synth.onUpdate()