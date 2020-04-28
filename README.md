# MSC_CognitiveRoboticsLab

Commands:
For each camera run one prediction model from here (https://github.com/microsoft/human-pose-estimation.pytorch). Adapting the topics in the code for the topics published by the cameras:

```
python pose_estimation/run.py --flip-test --cfg experiments/coco/resnet152/256x192_d256x3_adam_lr1e-3.yaml --model-file models/pytorch/pose_coco/pose_resnet_152_256x192.pth.tar
```
To add depth to the predicted images we need the following node (just one):
```
rosrun human_pose_multiview pose_generator_3d.py
```
The pose merger node can be launched by:
```
rosrun human_pose_multiview pose_fusion.py
```

For visualization:
Pointcloud of the images:
```
roslaunch human_pose_multiview pointcloud.launch
```
To draw poses in the images:
```
rosrun human_pose_drawer pose_drawer.py
```
