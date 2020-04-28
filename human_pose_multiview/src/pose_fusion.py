#!/usr/bin/env python

import rospy
import numpy as np
import math
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
from human_pose_multiview.msg import CustomMarkerArray
import tf
import tf2_ros
import tf2_geometry_msgs
import message_filters
import copy

# Publisher
FUSION_POSES_TOPIC = "/human_pose/pose3D/fusion"

FUSION_CHECK_CAM_1_POSES_TOPIC = "/human_pose/pose3D/fusion/check_cam_1"
FUSION_CHECK_CAM_2_POSES_TOPIC = "/human_pose/pose3D/fusion/check_cam_2"
FUSION_CHECK_CAM_4_POSES_TOPIC = "/human_pose/pose3D/fusion/check_cam_4"

# Subscribers
POSE_CAM_1_TOPIC = "/human_pose/pose3D/cam_1" 
POSE_CAM_2_TOPIC = "/human_pose/pose3D/cam_2" 
POSE_CAM_4_TOPIC = "/human_pose/pose3D/cam_4" 

# FLAGS
SHOW_CONFIDENCES = False
SHOW_COVARIANCES = True
SET_LIFETIME = False
USE_MAHALANOBIS = True
IDS_TO_VISUALIZE = [6,8,10,5,7,9] # Empty if all should be visualized
# "base" or "baxter_base"
BASE_FRAME = "baxter_base"

'''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
'''

class PoseFusion:

    def __init__(self):
        self.pose_cam_1_sub = message_filters.Subscriber(POSE_CAM_1_TOPIC, CustomMarkerArray)
        self.pose_cam_2_sub = message_filters.Subscriber(POSE_CAM_2_TOPIC, CustomMarkerArray)
        self.pose_cam_4_sub = message_filters.Subscriber(POSE_CAM_4_TOPIC, CustomMarkerArray)
    
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        
        self.pub = rospy.Publisher(FUSION_POSES_TOPIC, MarkerArray, queue_size=10)

        self.check_cam_1_pub = rospy.Publisher(FUSION_CHECK_CAM_1_POSES_TOPIC, MarkerArray, queue_size=10)
        self.check_cam_2_pub = rospy.Publisher(FUSION_CHECK_CAM_2_POSES_TOPIC, MarkerArray, queue_size=10)
        self.check_cam_4_pub = rospy.Publisher(FUSION_CHECK_CAM_4_POSES_TOPIC, MarkerArray, queue_size=10)
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.pose_cam_1_sub, self.pose_cam_2_sub, self.pose_cam_4_sub], 10, 0.1)
        self.sync.registerCallback(self.poses_callback)

        self.occluded_points_cam_2 = [4, 6, 8]
        self.occluded_points_cam_4 = [3, 5, 7]
        # Kalman filter parameters
        f = KalmanFilter(dim_x=6, dim_z=3)
        self.keypoints = 13
        self.initialized = [False for k in range(self.keypoints)]
        self.dt = 1/15.0	#15 fps

        f.inv = np.linalg.pinv
        # State transition matrix
        f.F = np.array([[1.,0.,0.,self.dt,0.,0.],
                        [0.,1.,0.,0.,self.dt,0.],
                        [0.,0.,1.,0.,0.,self.dt],
                        [0.,0.,0.,1.,0.,0.],
                        [0.,0.,0.,0.,1.,0.],
                        [0.,0.,0.,0.,0.,1.]])
        # Measurement function
        f.H = np.array([[1.,0.,0.,0.,0.,0.],
                        [0.,1.,0.,0.,0.,0.],
                        [0.,0.,1.,0.,0.,0.]])
        # Process uncertainty/noise
        f.Q = np.identity(6) * 1e-4
        
        # Measurement uncertainty/noise
        self.min_cov = 0.0025
        self.max_cov = 0.04
        self.steepness = 20
        self.steep_point = 0.6
        f.R = np.identity(3)*(self.min_cov+self.max_cov)/2
        f.R[2,2] = 0.05
        
        # Distance threshold
        if USE_MAHALANOBIS:
            self.distance_threshold = 60
        else:
            self.distance_threshold = 0.4
        # Covariance matrix
        f.P = np.identity(6) * 100
        
        # Create the KalmanFilter array
        self.kf_array = [copy.deepcopy(f) for kp in range(self.keypoints)]
        
        # Last-update time array
        self.time_since_update = [time.time() for kp in range(self.keypoints)]
        
        self.time_threshold = 0.25     # in secs
        
        # Auxiliar variables (for visualization)
        self.cam_1_check_covariances = [np.zeros((3,3)) for kp in range(self.keypoints)]
        self.cam_2_check_covariances = [np.zeros((3,3)) for kp in range(self.keypoints)]
        self.cam_4_check_covariances = [np.zeros((3,3)) for kp in range(self.keypoints)]
        
    ## Retrieves the transform to change from "number" optical frame to base
    def get_frame_transform(self, number):
        try:
            frame = 'cam_' + str(number) + '_color_optical_frame'
            return self.tfBuffer.lookup_transform(BASE_FRAME, frame, rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    ## Called everytime that there is 3Dpose information from the 3 cameras
    def poses_callback(self, pose_cam_1_data, pose_cam_2_data, pose_cam_4_data):
        #print(pose_cam_1_data)
        # Transform poses from local frame to base frame
        # Pose 1
        transformation = self.get_frame_transform(1)
        if transformation is None:
            return
        rot_cam_1 = self.quaternion_to_rotation_matrix(transformation.transform.rotation)
        pose_cam_1_transformed = []
        pose_cam_1_ids = []
        pose_cam_1_confidences = pose_cam_1_data.confidences
        for marker in pose_cam_1_data.markers:
            pose_cam_1_transformed.append(tf2_geometry_msgs.do_transform_pose(marker, transformation).pose)
            pose_cam_1_ids.append(marker.id)
        
        # Pose 2
        transformation = self.get_frame_transform(2)
        if transformation is None:
            return
        rot_cam_2 = self.quaternion_to_rotation_matrix(transformation.transform.rotation)
        pose_cam_2_transformed = []
        pose_cam_2_ids = []
        pose_cam_2_confidences = pose_cam_2_data.confidences
        for marker in pose_cam_2_data.markers:
            pose_cam_2_transformed.append(tf2_geometry_msgs.do_transform_pose(marker, transformation).pose)
            pose_cam_2_ids.append(marker.id)
        
        # Pose 3
        transformation = self.get_frame_transform(4)
        if transformation is None:
            return
        rot_cam_4 = self.quaternion_to_rotation_matrix(transformation.transform.rotation)
        pose_cam_4_transformed = []
        pose_cam_4_ids = []
        pose_cam_4_confidences = pose_cam_4_data.confidences
        for marker in pose_cam_4_data.markers:
            pose_cam_4_transformed.append(tf2_geometry_msgs.do_transform_pose(marker, transformation).pose)
            pose_cam_4_ids.append(marker.id)
        
        self.kalman_step(pose_cam_1_transformed, pose_cam_2_transformed, pose_cam_4_transformed, pose_cam_1_ids, pose_cam_2_ids, pose_cam_4_ids, pose_cam_1_confidences, pose_cam_2_confidences, pose_cam_4_confidences, rot_cam_1, rot_cam_2, rot_cam_4)
        self.publish_updated_poses(pose_cam_1_data.header.seq)
        
        # Publish poses to check if they are correct        
        self.publish_check_poses(pose_cam_1_data, pose_cam_2_data, pose_cam_4_data)
    
    ## Runs one kalman filter step for all joints (One KF on each)
    def kalman_step(self, pose_cam_1_transformed, pose_cam_2_transformed, pose_cam_4_transformed, pose_cam_1_ids, pose_cam_2_ids, pose_cam_4_ids, pose_cam_1_confidences, pose_cam_2_confidences, pose_cam_4_confidences, rot_cam_1, rot_cam_2, rot_cam_4):
        # Check if any kf needs to be initialized
        self.check_time_limits()
        
        # Initialize initial state
        if not all(self.initialized):
            for i, kf in enumerate(self.kf_array):
                if not self.initialized[i]:
                    if i in pose_cam_1_ids:
                        p = pose_cam_1_transformed[pose_cam_1_ids.index(i)].position
                        kf.x = np.array([p.x, p.y, p.z, 0, 0, 0]) 
                    elif i in pose_cam_2_ids:
                        p = pose_cam_2_transformed[pose_cam_2_ids.index(i)].position
                        kf.x = np.array([p.x, p.y, p.z, 0, 0, 0]) 
                    elif i in pose_cam_4_ids:
                        p = pose_cam_4_transformed[pose_cam_4_ids.index(i)].position
                        kf.x = np.array([p.x, p.y, p.z, 0, 0, 0]) 
                        
                    self.initialized[i] = True
                    self.time_since_update[i] = time.time()     

        for i, kf in enumerate(self.kf_array):
            if self.initialized[i]:
                kf.predict()
                if i in pose_cam_1_ids:
                    index = pose_cam_1_ids.index(i)
                    kf.R = self.update_measurement_noise(pose_cam_1_confidences[index])
                    kf.R = np.matmul(np.matmul(rot_cam_1,kf.R), rot_cam_1.T)
                    distance = self.calculate_distance(kf.x, kf.H, pose_cam_1_transformed[index], kf.P, kf.R)
                    if distance < self.distance_threshold:    
                        kf.update(self.pose_to_numpy(pose_cam_1_transformed[index]))
                        self.time_since_update[i] = time.time()
                        if SHOW_COVARIANCES:
                            if USE_MAHALANOBIS:
                                self.cam_1_check_covariances[i] = np.matmul(np.matmul(kf.H, kf.P), kf.H.T) + kf.R
                            else:
                                self.cam_1_check_covariances[i] = kf.R.copy()
                if i in pose_cam_2_ids:
                    index = pose_cam_2_ids.index(i)
                    kf.R = self.update_measurement_noise(pose_cam_2_confidences[index])
                    kf.R = np.matmul(np.matmul(rot_cam_2,kf.R), rot_cam_2.T)
                    distance = self.calculate_distance(kf.x, kf.H, pose_cam_2_transformed[index], kf.P, kf.R)
                    if distance < self.distance_threshold:
                        # Set R to max cov if the point is occluded
                        if i in self.occluded_points_cam_2:
                            kf.R[2,2] = self.max_cov 
                        kf.update(self.pose_to_numpy(pose_cam_2_transformed[index]))
                        self.time_since_update[i] = time.time()
                        if SHOW_COVARIANCES:
                            if USE_MAHALANOBIS:
                                self.cam_2_check_covariances[i] = np.matmul(np.matmul(kf.H, kf.P), kf.H.T) + kf.R
                            else:
                                self.cam_2_check_covariances[i] = kf.R.copy()
                if i in pose_cam_4_ids:
                    index = pose_cam_4_ids.index(i)
                    kf.R = self.update_measurement_noise(pose_cam_4_confidences[index])
                    kf.R = np.matmul(np.matmul(rot_cam_4,kf.R), rot_cam_4.T)
                    distance = self.calculate_distance(kf.x, kf.H, pose_cam_4_transformed[index], kf.P, kf.R)
                    if distance < self.distance_threshold:
                        # Set R to max cov if the point is occluded
                        if i in self.occluded_points_cam_4:
                            kf.R[2,2] = self.max_cov 
                        kf.update(self.pose_to_numpy(pose_cam_4_transformed[index]))
                        self.time_since_update[i] = time.time()
                        if SHOW_COVARIANCES:
                            if USE_MAHALANOBIS:
                                self.cam_4_check_covariances[i] = np.matmul(np.matmul(kf.H, kf.P), kf.H.T) + kf.R
                            else:
                                self.cam_4_check_covariances[i] = kf.R.copy()
    
    ## Checks the time stamps to reinitialize kalman filter
    def check_time_limits(self):
        t = time.time()
        for kf in range(self.keypoints):
            if (t - self.time_since_update[kf]) > self.time_threshold:
                #print('reinitialize point: ',kf)
                self.initialized[kf] = False
    
    ## Returns the distance between the measurement and the predicted measurement
    def measurement_distance(self, x, H, z):
        return np.linalg.norm(self.pose_to_numpy(z) - np.matmul(H,  x))
        
    def mahalanobis_distance(self, x, H, z, P, R):
        y = self.pose_to_numpy(z) - np.matmul(H,  x)
        S = np.matmul(np.matmul(H, P), H.T) + R
        d = np.matmul(np.matmul(y, np.linalg.pinv(S)), y.T) 
        rospy.loginfo(d)
        return d[0][0]

    def calculate_distance(self, x, H, z, P, R):
        if USE_MAHALANOBIS:
            return self.mahalanobis_distance(x, H, z, P, R)
        else:
            return self.measurement_distance(x, H, z)
    
    ## Converts quatertion to rotation matrix
    def quaternion_to_rotation_matrix(self, q):
        return np.array(R.from_quat([q.x, q.y, q.z, q.w]).as_dcm())
        
    
    ## Returns the new measurement noise matrix given the confidence of that point. Uses an sigmoid function  taking into account the steepest point, min and max covariance
    def update_measurement_noise(self, confidence):
        #value = self.min_cov + (self.max_cov - self.min_cov) * math.exp(-1 * self.scale_cov * confidence)
        value = (self.max_cov - self.min_cov) / (1 + math.exp(-self.steepness * ((1 - confidence) - (1 - self.steep_point)))) + self.min_cov
        return np.identity(3) * value
    
    ## Converts from Pose to numpy array
    def pose_to_numpy(self, pose):
        return np.array([pose.position.x, pose.position.y, pose.position.z]).reshape(1,3)
    
    ## Updates the color of fusion 3DPoses depending on the part of the body
    def update_color_of_parts(self, ma):
        head_ids = [0,1,2,3,4]
        left_arm_ids = [5,7,9]
        right_arm_ids = [6,8,10]
        left_leg_ids = [11,13,15]
        right_leg_ids = [12,14,16]
        
        for marker in ma.markers:
            if marker.id in head_ids:
                marker.color.r = 1
                marker.color.g = 0.12
                marker.color.b = 0.05
            elif marker.id in left_arm_ids:
                marker.color.r = 0.9
                marker.color.g = 0.25
                marker.color.b = 0.05
            elif marker.id in right_arm_ids:
                marker.color.r = 1
                marker.color.g = 0.66
                marker.color.b = 0.05
            elif marker.id in left_leg_ids:
                marker.color.r = 0.5
                marker.color.g = 0.48
                marker.color.b = 0
            elif marker.id in right_leg_ids:
                marker.color.r = 1
                marker.color.g = 0.97
                marker.color.b = 0
            marker.color.a = 1 if marker.id in IDS_TO_VISUALIZE or not IDS_TO_VISUALIZE else 0 # Invisible if not selected
                
        return ma
    
    ## Creates a marker given a point, an index and a header
    def create_marker(self, p, i, h):
        marker = Marker()
        marker.header = h
        marker.pose.position.x = p[0]
        marker.pose.position.y = p[1]
        marker.pose.position.z = p[2]
        marker.pose.orientation.w = 1
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        marker.ns = "basic_shape"
        marker.id = i
        marker.lifetime = rospy.Duration(0.5) if SET_LIFETIME else rospy.Duration(0)
        marker.type = Marker.SPHERE
        
        return marker
    
    ## Publishes fusion 3DPose
    def publish_updated_poses(self, seq):
        ma = MarkerArray()
        h = Header(frame_id=BASE_FRAME, seq=seq)

        points = []

        for i, kf in enumerate(self.kf_array):
            marker = self.create_marker(kf.x, i, h)
            ma.markers.append(marker)
            
        ma = self.update_color_of_parts(ma)
        self.pub.publish(ma)
        
    ## Prepares properties of markers to be published
    def set_check_poses_markers(self, pose_markers, r,g,b):
        for marker in pose_markers:
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1 if marker.id in IDS_TO_VISUALIZE or not IDS_TO_VISUALIZE else 0 # Invisible if not selected
            marker.type = Marker.SPHERE
            marker.lifetime = rospy.Duration(0.5) if SET_LIFETIME else rospy.Duration(0)
        
        return pose_markers
    
    ## Publishes the 3D poses from each camera
    def publish_check_poses(self, pose_cam_1_data, pose_cam_2_data, pose_cam_4_data):
        ma_1 = MarkerArray()
        ma_1.markers = pose_cam_1_data.markers
        ma_1.markers = self.set_check_poses_markers(ma_1.markers, 0.5, 0.61, 0.9)
        if SHOW_CONFIDENCES:
            ma_1 = self.add_confidence_markers(ma_1, pose_cam_1_data.confidences)
        elif SHOW_COVARIANCES:
            ma_1 = self.add_covariance_markers(ma_1, 1)
        self.check_cam_1_pub.publish(ma_1)
        
        ma_2 = MarkerArray()
        ma_2.markers = pose_cam_2_data.markers
        ma_2.markers = self.set_check_poses_markers(ma_2.markers, 160/255.0, 1.0, 121/255.0)
        if SHOW_CONFIDENCES:
            ma_2 = self.add_confidence_markers(ma_2, pose_cam_2_data.confidences)
        elif SHOW_COVARIANCES:
            ma_2 = self.add_covariance_markers(ma_2, 2)
        self.check_cam_2_pub.publish(ma_2)
                
        ma_4 = MarkerArray()
        ma_4.markers = pose_cam_4_data.markers
        ma_4.markers = self.set_check_poses_markers(ma_4.markers, 0.5, 0.82, 0.9)
        if SHOW_CONFIDENCES:
            ma_4 = self.add_confidence_markers(ma_4, pose_cam_4_data.confidences)
        elif SHOW_COVARIANCES:
            ma_4 = self.add_covariance_markers(ma_4, 4)
        self.check_cam_4_pub.publish(ma_4)
    
    ## Add markers representing the covariance of each point
    def add_covariance_markers(self, ma, camera):
        new_ma = copy.deepcopy(ma)
        if USE_MAHALANOBIS:
            for marker in ma.markers:
                if marker.id < self.keypoints and (marker.id in IDS_TO_VISUALIZE or not IDS_TO_VISUALIZE):
                    cov = self.cam_2_check_covariances[marker.id] if camera == 2 else self.cam_4_check_covariances[marker.id] 
                    cov_marker = copy.deepcopy(marker)
                    cov_marker.type = Marker.SPHERE
                    w, v = np.linalg.eigh(cov)   # Eigen-values and Eigen-vectors
                    cov_marker.scale.x = 2 * math.sqrt(w[0])
                    cov_marker.scale.y = 2 * math.sqrt(w[1])
                    cov_marker.scale.z = 2 * math.sqrt(w[2])
                    cov_marker.color.a = 0.3
                    cov_marker.ns = "covariances"

                    rotation = R.from_dcm(v).as_quat()

                    cov_marker.pose.orientation.w = rotation[0]
                    cov_marker.pose.orientation.x = rotation[1]
                    cov_marker.pose.orientation.y = rotation[2]
                    cov_marker.pose.orientation.z = rotation[3]
                    
                    new_ma.markers.append(cov_marker)
        else:
            for marker in ma.markers:
                if marker.id < self.keypoints and (marker.id in IDS_TO_VISUALIZE or not IDS_TO_VISUALIZE):
                    cov = self.cam_2_check_covariances[marker.id] if camera == 2 else self.cam_4_check_covariances[marker.id]
                    cov_marker = copy.deepcopy(marker)
                    cov_marker.type = Marker.SPHERE
                    cov_marker.scale.x = 2 * math.sqrt(cov[0,0])
                    cov_marker.scale.y = 2 * math.sqrt(cov[1,1])
                    cov_marker.scale.z = 2 * math.sqrt(cov[2,2])
                    cov_marker.color.a = 0.3
                    cov_marker.ns = "covariances"
                    
                    new_ma.markers.append(cov_marker)
        return new_ma
    
    ## Add markers with the confidence of each point as text
    def add_confidence_markers(self, ma, confidences):
        for i, c in enumerate(confidences):
            marker = copy.deepcopy(ma.markers[i])
            marker.type = Marker.TEXT_VIEW_FACING
            marker.text = str(round(c, 2))
            marker.pose.position.y -= 0.08
            marker.ns = "confidence"
            
            ma.markers.append(marker)
        return ma
        
if __name__ == '__main__':
    rospy.init_node('pose_fusion', anonymous=True)

    pose_fusion = PoseFusion()

    rospy.spin()
