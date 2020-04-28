#!/usr/bin/env python

import rospy
import time
import cv2
import sys
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from openpose_ros_msgs.msg import OpenPoseHumanList
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from human_pose_multiview.msg import CustomMarkerArray


CAMERA = str(sys.argv[1])   # Read camera from parameters

# Subscribers
HUMAN_POSE_TOPIC = "/human_pose_estimation/human_list/cam_" + CAMERA
COLOR_IMAGE_TOPIC = "/cam_" + CAMERA + "/color/image_raw"
DEPTH_IMAGE_TOPIC = "/cam_" + CAMERA + "/depth_registered/image_rect"
CAMERA_PARAMETERS = "/cam_" + CAMERA + "/color/camera_info"

# Publisher
HUMAN_POSE_DRAWER_TOPIC = "/human_pose/pose3D/cam_" + CAMERA

class PoseGenerator:
    def __init__(self, pub, K):
        self.bridge = CvBridge()
        #self.pose_sub = rospy.Subscriber(HUMAN_POSE_TOPIC, MarkerArray, self.pose_callback, queue_size=10)
        #self.image_sub = rospy.Subscriber(COLOR_IMAGE_TOPIC, Image, self.image_callback, queue_size=10)
        self.pose_sub = message_filters.Subscriber(HUMAN_POSE_TOPIC, OpenPoseHumanList)
        self.image_sub = message_filters.Subscriber(COLOR_IMAGE_TOPIC, Image)
        self.depth_sub = message_filters.Subscriber(DEPTH_IMAGE_TOPIC, Image)

        self.sync = message_filters.ApproximateTimeSynchronizer([self.pose_sub, self.image_sub, self.depth_sub], 10, 0.1)
        self.sync.registerCallback(self.callback)
        
        self.pub = pub
        self.K = K
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]
        
        self.depth_env_delta = 2
        self.min_depth = 1000
        self.max_depth = 2500
        
        self.threshold = 0.25
        self.g_depth_scale = 1000.0

        
    def callback(self, data_pose, data_image, data_depth):
        #print("message \t - \t", data_image.header.seq, data_pose.header.seq)
        cv_color = self.bridge.imgmsg_to_cv2(data_image, "rgb8")
        cv_depth = self.bridge.imgmsg_to_cv2(data_depth, "16UC1")

        ma = CustomMarkerArray()
        h = Header(frame_id=data_image.header.frame_id, seq=data_image.header.seq)
        ma.header = h
        
        for kp_idx, point in enumerate(data_pose.human_list[0].body_key_points_with_prob):
                if point.prob >= self.threshold:
                    u, v = int(round(point.x)), int(round(point.y))
                    depth_candidates = []
                    for uu in range(u-self.depth_env_delta, u+self.depth_env_delta):
                        for vv in range(v-self.depth_env_delta, v+self.depth_env_delta):
                            if uu<0 or vv<0 or uu>=cv_color.shape[1] or vv>=cv_color.shape[0]:
                                break
                            if cv_depth[vv,uu] > self.min_depth and cv_depth[vv,uu] < self.max_depth:
                                depth_candidates.append(cv_depth[vv,uu])
                    
                    if not depth_candidates:
                        break
                      
                    depth_candidates.sort()

                    #z = 0
                    #if kp_idx==11 or kp_idx==12:
                        #z = depth_candidates[-1] / self.g_depth_scale
                    #else:
                    z = depth_candidates[len(depth_candidates)/2] / self.g_depth_scale

                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    
                    marker = Marker()
                    marker.header = h
                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.position.z = z
                    marker.pose.orientation.w = 1
                    marker.pose.orientation.x = 0
                    marker.pose.orientation.y = 0
                    marker.pose.orientation.z = 0
                                    
                    marker.scale.x = 0.05
                    marker.scale.y = 0.05
                    marker.scale.z = 0.05
                    
                    marker.ns = "joints"
                    marker.id = kp_idx
                    marker.color.r = 1.0
                    marker.color.a = 1.0
                    
                    ma.confidences.append(point.prob)
                    ma.markers.append(marker)
        
        self.pub.publish(ma)

class CameraCalibSubscriber():
    def __init__(self):
        self.subscriber = rospy.Subscriber(CAMERA_PARAMETERS,
                                        CameraInfo, self.camera_callback, queue_size=1)
        self.stop = False
        self.K = None
        self.camera_frame_id = None
        self.camera_seq = None

    def camera_callback(self, data):
        self.K = data.K
        self.camera_frame_id = data.header.frame_id
        self.camera_seq = data.header.seq
        self.stop = True

    def wait_for_calib(self):
        try:
            while not self.stop:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down")

        return self.K, self.camera_frame_id, self.camera_seq

if __name__ == '__main__':
    rospy.init_node('pose_drawer_3d', anonymous=True)
    
    #read calib from ros topic
    camera_calib = CameraCalibSubscriber()
    K, camera_frame_id, camera_seq = camera_calib.wait_for_calib()
    
    
    pub = rospy.Publisher(HUMAN_POSE_DRAWER_TOPIC, CustomMarkerArray, queue_size=10)
    human_pose_drawer = PoseGenerator(pub, K)

    rospy.spin()
    
    
    
    
