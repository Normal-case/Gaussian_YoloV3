import argparse
import yaml
import os
import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
from utils.vis_bbox import vis_bbox
import matplotlib.pyplot as plt

# Choose config file for this demo
cfg_path = './config/gaussian_yolov3_eval.cfg'
# Specify checkpoint file which contains the weight of the model you want to use
ckpt_path = './checkpoints/snapshot50000.ckpt'
# Path to the image file fo the demo
image_path = './demo/image/'
# Detection threshold
detect_thresh = 0.5
# Use CPU if gpu < 0 else use GPU
gpu = -1

# Load configratio parameters for this demo
with open(cfg_path, 'r') as f:
    cfg = yaml.load(f)

model_config = cfg['MODEL']
imgsize = cfg['TEST']['IMGSIZE']
confthre = cfg['TEST']['CONFTHRE'] 
nmsthre = cfg['TEST']['NMSTHRE']
gaussian = cfg['MODEL']['GAUSSIAN']

# if detect_thresh is not specified, the parameter defined in config file is used
if detect_thresh:
    confthre = detect_thresh

# Load model
model = YOLOv3(model_config)

# Load weight from the checkpoint
print("loading checkpoint %s" % (ckpt_path))
state = torch.load(ckpt_path)

if 'model_state_dict' in state.keys():
    model.load_state_dict(state['model_state_dict'])
else:
    model.load_state_dict(state)

model.eval()

if gpu >= 0:
    # Send model to GPU
    model.cuda()

# Load img list
external_list = os.listdir(image_path)

# split
length = 50
split_list = []
index = 0
for i in range(20):
  split_list.append(external_list[index:index+length])
  index += length
NoObject = []

# train Inference
for j in range(20):
  for i in range(length):
    img = cv2.imread(image_path + split_list[j][i])
    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
    img = np.transpose(img / 255., (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)

    if gpu >= 0:
      # Send model to GPU
      img = Variable(img.type(torch.cuda.FloatTensor))
    else:
      img = Variable(img.type(torch.FloatTensor))

    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, 65, confthre, nmsthre)

    if outputs[0] is None:
        print("No Objects Deteted!!")
        print('img:', split_list[j][i])
        NoObject.append(split_list[j][i])
        continue

    # Visualize detected bboxes
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

    bboxes = list()
    classes = list()
    scores = list()
    colors = list()
    sigmas = list()
    world = list()
    dist = list()

    for Output in outputs[0]:
        x1, y1, x2, y2, conf, cls_conf, cls_pred = Output[:7]
        if gaussian:
            sigma_x, sigma_y, sigma_w, sigma_h = Output[7:]
            sigmas.append([sigma_x, sigma_y, sigma_w, sigma_h])

        cls_id = coco_class_ids[int(cls_pred)]
        box = yolobox2label([y1, x1, y2, x2], info_img)
        bboxes.append(box)
        classes.append(cls_id)
        scores.append(cls_conf * conf)
        colors.append(coco_class_colors[int(cls_pred)])
        
        # image size scale used for sigma visualization
        h, w, nh, nw, _, _ = info_img
        sigma_scale_img = (w / nw, h / nh)

        # calculate distance and x, y, z
        height = box[2].item() - box[0].item()
        width = box[3].item() - box[1].item()
        x_cent = box[1].item() + 1/2 * width
        y_cent = box[0].item() + 1/2 * height
        inverse = [1, 3]
        if cls_id in inverse:
            # 역삼각형 표지판
            z_world = 800 / width * 1060
        elif cls_id == 65:
            # 신호등
            z_world = 1065 / width * 1060
        else:
            # 나머지
            z_world = 1065 / width * 1060
        x_world = (x_cent - 960) / 1060 * z_world
        y_world = (y_cent - 600) / 1060 * z_world
        distance = round(((x_world**2) + (y_world**2) + (z_world**2))**0.5 / 1000, 2)
        world.append((x_world, y_world, z_world))
        dist.append(distance)
    
    fig, ax = vis_bbox(
        img_raw, bboxes, label=classes, score=scores, label_names=coco_class_names, sigma=sigmas, 
        sigma_scale_img=sigma_scale_img, world = world, distance = dist,
        sigma_scale_xy=2., sigma_scale_wh=2.,  # 2-sigma
        show_inner_bound=False,  # do not show inner rectangle for simplicity
        instance_colors=colors, linewidth=3)

    fig.savefig('./demo/result/' + split_list[j][i])
    print('{} save'.format(split_list[j][i]))