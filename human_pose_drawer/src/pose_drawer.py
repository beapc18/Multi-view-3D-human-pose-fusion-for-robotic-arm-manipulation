#!/usr/bin/env python

from Camera import * 
import time
import cv2

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from openpose_ros_msgs.msg import OpenPoseHumanList
from visualization_msgs.msg import MarkerArray, Marker
from human_pose_multiview.msg import CustomMarkerArray

import tf
import tf2_ros
import tf2_geometry_msgs

# Subscribers
FUSION_POSES_TOPIC = "/human_pose/pose3D/fusion"

CAM_1_POSE_TOPIC = "/human_pose_estimation/human_list/cam_1"
CAM_2_POSE_TOPIC = "/human_pose_estimation/human_list/cam_2" 
CAM_4_POSE_TOPIC = "/human_pose_estimation/human_list/cam_4" 

CAM_1_COLOR_IMAGE_TOPIC = "/cam_1/color/image_raw"
CAM_2_COLOR_IMAGE_TOPIC = "/cam_2/color/image_raw"
CAM_4_COLOR_IMAGE_TOPIC = "/cam_4/color/image_raw"

CAM_1_INFO_TOPIC = "/cam_1/color/camera_info"
CAM_2_INFO_TOPIC = "/cam_2/color/camera_info" 
CAM_4_INFO_TOPIC = "/cam_4/color/camera_info" 

# Publishers
CAM_1_POSE_DRAWER_TOPIC = "/cam_1/pose_projected"
CAM_2_POSE_DRAWER_TOPIC = "/cam_2/pose_projected"
CAM_4_POSE_DRAWER_TOPIC = "/cam_4/pose_projected"

# Flags
SHOW_ONLY_FUSION = True


class HumanPoseDrawer:
    def __init__(self, cam_1, cam_2, cam_4, cam_1_transformation, cam_2_transformation, cam_4_transformation):
        self.bridge = CvBridge()
        
        self.cam_1 = cam_1
        self.cam_2 = cam_2
        self.cam_4 = cam_4
        
        self.cam_1_transformation = cam_1_transformation
        self.cam_2_transformation = cam_2_transformation
        self.cam_4_transformation = cam_4_transformation
        
        self.pose_fusion_sub = message_filters.Subscriber(FUSION_POSES_TOPIC, MarkerArray)
        
        self.pose_cam_1_sub = message_filters.Subscriber(CAM_1_POSE_TOPIC, OpenPoseHumanList)
        self.pose_cam_2_sub = message_filters.Subscriber(CAM_2_POSE_TOPIC, OpenPoseHumanList)
        self.pose_cam_4_sub = message_filters.Subscriber(CAM_4_POSE_TOPIC, OpenPoseHumanList)
        
        self.cam_1_color_sub = message_filters.Subscriber(CAM_1_COLOR_IMAGE_TOPIC, Image)
        self.cam_2_color_sub = message_filters.Subscriber(CAM_2_COLOR_IMAGE_TOPIC, Image)
        self.cam_4_color_sub = message_filters.Subscriber(CAM_4_COLOR_IMAGE_TOPIC, Image)
        
        self.sync_cam_1 = message_filters.ApproximateTimeSynchronizer([self.pose_fusion_sub, self.pose_cam_1_sub, self.cam_1_color_sub], 10, 0.1, allow_headerless=True)
        self.sync_cam_1.registerCallback(self.callback)
        
        self.sync_cam_2 = message_filters.ApproximateTimeSynchronizer([self.pose_fusion_sub, self.pose_cam_2_sub, self.cam_2_color_sub], 10, 0.1, allow_headerless=True)
        self.sync_cam_2.registerCallback(self.callback)
        
        self.sync_cam_4 = message_filters.ApproximateTimeSynchronizer([self.pose_fusion_sub, self.pose_cam_4_sub, self.cam_4_color_sub], 10, 0.1, allow_headerless=True)
        self.sync_cam_4.registerCallback(self.callback)
        
        self.cam_1_pub = rospy.Publisher(CAM_1_POSE_DRAWER_TOPIC, Image, queue_size=10)
        self.cam_2_pub = rospy.Publisher(CAM_2_POSE_DRAWER_TOPIC, Image, queue_size=10)
        self.cam_4_pub = rospy.Publisher(CAM_4_POSE_DRAWER_TOPIC, Image, queue_size=10)
                
    def callback(self, data_fusion, data_pose, data_image):
        #print("message \t - \t", data_image.header.seq, data_pose.header.seq)
        cv_image = self.bridge.imgmsg_to_cv2(data_image, "rgb8")
        frame = data_image.header.frame_id
        
        camera = None
        pub = None
        transforamtion = None
        
        if frame == "cam_1_color_optical_frame":
            camera = self.cam_1
            pub = self.cam_1_pub
            transformation = self.cam_1_transformation
        elif frame == "cam_2_color_optical_frame":
            camera = self.cam_2
            pub = self.cam_2_pub
            transformation = self.cam_2_transformation
        elif frame == "cam_4_color_optical_frame":
            camera = self.cam_4
            pub = self.cam_4_pub
            transformation = self.cam_4_transformation
        
        if not SHOW_ONLY_FUSION:
            # Draw camera 2D points
            for point in data_pose.human_list[0].body_key_points_with_prob:
                if point.prob >= 0.25:
                    cv_image = cv2.circle(cv_image, (int(point.x), int(point.y)), 10, (0,255,0), -1)
        
        # Transform the 3dpose from base frame to the correct camera frame
        for marker in data_fusion.markers[:-4]:
            transformed_fusion = tf2_geometry_msgs.do_transform_pose(marker, transformation).pose
            
            fusion_projected = camera.project(self.pose_to_numpy(transformed_fusion))[0]
            cv_image = cv2.circle(cv_image, (int(fusion_projected[0]), int(fusion_projected[1])), 10, (marker.color.r * 255, marker.color.g * 255, marker.color.b * 255), -1)
        #print(transformed_fusion)
        ## Draw projected fusion pose
        #for pose in transformed_fusion:
            #fusion_projected = camera.project(self.pose_to_numpy(pose))[0]
            #cv_image = cv2.circle(cv_image, (int(fusion_projected[0]), int(fusion_projected[1])), 5, (255,0,0), -1)
                    
        pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "rgb8"))
    
    
    def pose_to_numpy(self, pose):
        return np.array([pose.position.x, pose.position.y, pose.position.z])

class CameraCalibSubscriber:
    def __init__(self, topic):
        self.subscriber = rospy.Subscriber(topic,
                                        CameraInfo, self.camera_callback, queue_size=1)
        self.stop = False
        self.K = None
        self.camera_frame_id = None

    def camera_callback(self, data):
        self.K = np.reshape(np.array(data.K), [3, 3])
        self.camera_frame_id = data.header.frame_id
        self.stop = True

    def wait_for_calib(self):
        try:
            while not self.stop:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down")

        return self.K, self.camera_frame_id



if __name__ == '__main__':
    #read calib from ros topic
    rospy.init_node('pose_drawer', anonymous=True)
    
    # Get transforms before calibration
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
    cam_1_transformation = None
    cam_2_transformation = None
    cam_4_transformation = None
    
    print("Looking for transforms!")
    while cam_1_transformation is None or cam_2_transformation is None or cam_4_transformation is None:       
        try:
            cam_1_transformation = tfBuffer.lookup_transform("cam_1_color_optical_frame", 'base', rospy.Time(0))
            cam_2_transformation = tfBuffer.lookup_transform("cam_2_color_optical_frame", 'base', rospy.Time(0))
            cam_4_transformation = tfBuffer.lookup_transform("cam_4_color_optical_frame", 'base', rospy.Time(0))
            if not cam_1_transformation is None and not cam_2_transformation is None and not cam_4_transformation is None:
                break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException), e:
            continue
    print("All transforms received!")
    
    # Camera 1
    cam_1_calib = CameraCalibSubscriber(CAM_1_INFO_TOPIC)
    K_1, cam_1_frame_id = cam_1_calib.wait_for_calib()
    cam_1 = Camera(K_1)
    
    # Camera 2
    cam_2_calib = CameraCalibSubscriber(CAM_2_INFO_TOPIC)
    K_2, cam_2_frame_id = cam_2_calib.wait_for_calib()
    cam_2 = Camera(K_2)
    
    # Camera 4
    cam_4_calib = CameraCalibSubscriber(CAM_4_INFO_TOPIC)
    K_4, cam_4_frame_id = cam_4_calib.wait_for_calib()
    cam_4 = Camera(K_4)

    human_pose_drawer = HumanPoseDrawer(cam_1, cam_2, cam_4, cam_1_transformation, cam_2_transformation, cam_4_transformation)

    rospy.spin()
    
    
    
    
