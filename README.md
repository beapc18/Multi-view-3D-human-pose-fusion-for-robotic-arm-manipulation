# Multi-view 3D human pose fusion for robotic arm manipulation

The **goal** of this project is to implement a **real-time system** that allows a **human operator to control an articulated robot by recognizing body poses** and extracting movement information from them. The system estimates the operator’s 3D pose from **several RGBD cameras** and merges them into one central representation. From that fused **3D pose**, information about the position and orientation of specific joints can be extracted. In our case, we show that the system allows the operator to control a **robotic arm** by moving its right arm.

In the original setup of this project, the operator is situated in front of the robot where there are three RGB-D Realsense cameras and these cameras have different perspectives of the environment.

<p align="center">
<img src="https://github.com/beapc18/MSC_CognitiveRoboticsLab/blob/master/images/baxter_setup.png" width="40%"> 
</p>

<p align="center">
<img src="https://github.com/beapc18/MSC_CognitiveRoboticsLab/blob/master/images/camera_perspectives.png" width="80%">
</p>

The developed modules of our project interact as follows:

<p align="center">
<img src="https://github.com/beapc18/MSC_CognitiveRoboticsLab/blob/master/images/experiment_scheme.png" width="100%">  
</p>

For the **extraction of human poses**, we use [Microsoft Human Pose Estimation](https://github.com/microsoft/human-pose-estimation.pytorch).
Since this model only estimates the 2D position of the pose joints in the image, the depth images from the same cameras are used additional to obtain the third coordinate, by calculating the median
depth value of the surrounding pixels and using the pinhole camera model.

After the 3D predictions from the cameras are obtained, they are fused into one central prediction. For the **pose fusion** process, one **Kalman Filter** for each joint is used. Each one of them estimates the current pose of its joint based on the estimations made for each one of the three cameras.

<p align="center">
<img src="https://github.com/beapc18/MSC_CognitiveRoboticsLab/blob/master/images/pose_fusion.gif" width="80%">  
</p>

## Experiment

In the experiment, a [UR5 robotic arm](https://www.universal-robots.com/) situated in front of the operator moves replicating the operator’s right arm movement. The operator’s arm pose is calculated in real-time using the system previously explained. A trajectory is generated from sequential fused poses of the arm, and is transmitted to the robot arm controller in order for it to replicate the movement.

<p align="center">
<img src="https://github.com/beapc18/MSC_CognitiveRoboticsLab/blob/master/images/arm_with_orientation.gif" width="80%">  
</p>


## Setup guide

### Requirements
- Python 3
- ROS melodic
- Python-rospy

### Prepare ROS workspace
You need to initialize your ROS workspace as explained in the [ROS tutorial](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment) and clone the Git repository into your workspace.

### Launch ROS nodes
For each camera you need to run one prediction model from here [Microsoft Human Pose Estimation](https://github.com/microsoft/human-pose-estimation.pytorch) and adapts the topics in the code for the topics published by the cameras. The contents of the folder pose_estimation must be placed in the pose_estimation folder of the prediction microsoft model:
```
python pose_estimation/run.py --flip-test --cfg experiments/coco/resnet152/256x192_d256x3_adam_lr1e-3.yaml --model-file models/pytorch/pose_coco/pose_resnet_152_256x192.pth.tar
```

In order to add depth to the predicted images, we need the following node (just one):
```
rosrun human_pose_multiview pose_generator_3d.py
```

The pose merger node can be launched by:
```
rosrun human_pose_multiview pose_fusion.py
```

### Visualization
Pointcloud of the images:
```
roslaunch human_pose_multiview pointcloud.launch
```
Draw poses in the images:
```
rosrun human_pose_drawer pose_drawer.py
```
