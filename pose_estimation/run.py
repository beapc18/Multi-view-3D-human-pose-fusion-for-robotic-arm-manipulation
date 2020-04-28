from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np
import cv2
import math
import time

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.inference import get_max_preds
from utils.transforms import flip_back
from utils.utils import create_logger

import dataset
import models

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

COCO_num_joints = 17
COCO_skeleton = [[15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12],[5,6],[5,7],
        [6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[2,4],[3,5],[4,6]]

COCO_colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
              (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
              (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file



from torch.utils.cpp_extension import load

ros_bridge = load(
    name="ros_bridge",
    sources=["pose_estimation/ros_bridge.cpp"],
    extra_include_paths=["/opt/ros/melodic/include", "/usr/include/eigen3", "/home/lab20/cogrob/devel/include/"],
    extra_ldflags=["-L/opt/ros/melodic/lib", "-lroscpp", "-leigen_conversions" ],
    extra_cflags=['-O3'],
    verbose=True,
)

def quaternion_average(q, weights=None):
    """
    Method by https://arc.aiaa.org/doi/abs/10.2514/1.28949
    """

    q = q * weights.sqrt().unsqueeze(0)

    dev = q.device

    M = torch.matmul(q, q.t()).cpu()
    eig_val, eig_vec = torch.symeig(M, eigenvectors=True)
    _, idx = eig_val[:].max(dim=0)
    q_nasa = eig_vec[:,idx]

    return q_nasa.to(dev)

if __name__ == "__main__":

    args = parse_args()
    reset_config(config, args)
    
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    ros_bridge.init()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )

    if config.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        print('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
        
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
    else:
        print("WARNING: Not using CUDA!")
        device = torch.device('cpu')    

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    model.eval()
    torch.set_grad_enabled(False)

    flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    
    print('Model Input size: {}'.format(config.MODEL.IMAGE_SIZE))
    
    #img_num = 0
    while True:
        #start = time.time()
        rgb, msg_holder = ros_bridge.wait_for_frame()
        rgb = rgb.to(device)
        rgb = rgb.permute(2,0,1) # hwc -> chw
        rgb = rgb.unsqueeze(0) # -> nchw  
        
        permute = [2, 1, 0]
        bgr = rgb[:, permute] # rgb -> bgr
        im_height = rgb.shape[2]
        im_width = rgb.shape[3]
        #print('Input image: {}x{}'.format(im_width, im_height))            
        # crop to 4:3 (width:height) format, using full height (TODO: actually, the network was trained for upright crops of persons (height=4:width=3)...)
        im_crop_width = int(im_height * 4.0 / 3.0)
        im_crop_x0 = (im_width - im_crop_width) // 2 #crop center of image
        
        cropped = bgr[:, :, :, im_crop_x0:im_crop_x0 + im_crop_width]
        resized = torch.nn.functional.interpolate(cropped.float(), size=(config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]), mode='area')

        img = resized / 255.0
        input = normalize(img[0]).unsqueeze(0)
            
        # compute output
        output = model(input)
        
        if config.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
            input_flipped = np.flip(input.cpu().numpy(), 3).copy()
            input_flipped = torch.from_numpy(input_flipped).cuda()
            output_flipped = model(input_flipped)
            output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if config.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
                # output_flipped[:, :, :, 0] = 0

            output = (output + output_flipped) * 0.5
            
        batch_heatmaps = output.clone().cpu().numpy()
        coords, maxvals = get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # post-processing
        if config.TEST.POST_PROCESS:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = batch_heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                        diff = np.array([hm[py][px+1] - hm[py][px-1],
                                          hm[py+1][px]-hm[py-1][px]])
                        coords[n][p] += np.sign(diff) * .25

        preds = coords.copy()
        preds = preds[0] # batchsize = 1
        maxvals = maxvals[0]
            
        factor_in_out = float(resized.shape[2]) / heatmap_height
        factor_orig_out = float(cropped.shape[2]) / heatmap_height
        #print('Factor in-out: {}, factor orig-out: {}'.format(factor_in_out, factor_orig_out))
            
        preds *= factor_orig_out
        preds[:, 0] += im_crop_x0 # add x-offset
        
        preds_tensor = torch.from_numpy(preds)
        maxvals_tensor = torch.from_numpy(maxvals)
        
        ros_bridge.publish_pose(preds_tensor, maxvals_tensor, msg_holder)
        
        ##draw skeleton
        #centers = {}
        #img_res = bgr[0].cpu().permute(1,2,0).numpy()
        #for i in range(COCO_num_joints):
            #if (preds[i][0] == im_crop_x0 and preds[i][1] == 0.0) or maxvals[i] < 0.25: # for no detection, preds[i] will be equal to [im_crop_x0, 0.0]
                #continue

            #body_part = preds[i]
            #center = (int(body_part[0] + 0.5), int(body_part[1] + 0.5)) # round to neares integer
            #centers[i] = center
            #img_res = cv2.circle(img_res, center, 3, COCO_colors[i], thickness=3, lineType=8, shift=0)

        ## draw line
        #for pair in COCO_skeleton:
            #if pair[0] not in centers.keys() or pair[1] not in centers.keys():
                #continue

            #img_res = cv2.line(img_res, centers[pair[0]], centers[pair[1]], COCO_colors[pair[0]], 3)
                
        #out_path_skeleton = '/home/baxter/Pictures/baxter_obj_test_pose/frame{}_skeleton.jpg'.format(img_num)
        #out_path_resized = '/home/baxter/Pictures/baxter_obj_test_pose/frame{}_resized.jpg'.format(img_num)
        #cv2.imwrite(out_path_skeleton, img_res)
        #cv2.imwrite(out_path_resized, resized.byte()[0].permute(1,2,0).cpu().numpy())
        #img_num += 1
        
        #end = time.time()
        #remain = start + loop_delay - end
        #if remain > 0:
        #    time.sleep(remain)
