import torch
import cv2
import queue
import argparse
import sys
import threading
from tqdm import tqdm, trange
import os
import math
import glob
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.path as mpltPath

sys.path.insert(1, 'yolov5-master/')
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

def draw_rectangle(img, x1, x2, y1, y2, color): # top-left and bottom-right coordinates of rectangle
    box_label = ''
    if id == 'vps':
        box_label = 'vps'
    elif id == 'helmet':
        box_label = 'helmet'
    color_list = (0, 255, 0)
    if color == 'Blue':
        color_list = (255, 0, 0)
    elif color == 'Red':
        color_list = (0, 0, 255)
        if id == 'helmet':
            color_list = (0, 127, 127)
    cv2.rectangle(img, (x1, y1), (x2, y2), color_list, 10)
    # labelSize = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)
    _x1 = x1 # bottomleft x of text
    _y1 = y1 # bottomleft y of text
    _x2 = x1 #+labelSize[0][0] # topright x of text
    _y2 = y1 #-labelSize[0][1] # topright y of text
    cv2.rectangle(img, (_x1,_y1), (_x2,_y2), color_list, cv2.FILLED)
    # cv2.putText(img, box_label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
    return

img_src = '/app/1028_frame/'
# img_src = glob.glob(imgs_src)
# print(imgs_src)

# img_src = check_file(img_src)
# print(f'check_file = {img_src}')
imgsz = (640, 640)

device = torch.device('cuda')
data = 'yolov5-master/data/cone.yaml'
# weights = 'Models/cone_model_0705.pt'
weights = 'Models/cone_model_0527.pt'
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
# print(stride, names, pt)
imgsz = check_img_size(imgsz, s=stride)  # check image size
# print(imgsz)
dataset = LoadImages(img_src, img_size=imgsz, stride=stride, auto=pt)
bs = 1  # batch_size
vid_path, vid_writer = [None] * bs, [None] * bs
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
model.eval()

conf_thres=0.7
iou_thres=0.45
classes=None
agnostic_nms=False
max_det=1000
'''
test_img = None
dt = (Profile(), Profile(), Profile())

# video cap
# out = cv2.VideoWriter('results/1028_1203.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (1920, 1080))

for path, im, im0s, _, _ in dataset:
    # print(f'This is img path = {path}')
    with dt[0]:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        if test_img is None:
            test_img = im
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    pred_ = pred[0].detach().cpu().numpy()
    # print(pred_.shape)
    # print('Before:')
    # print(pred)
    xywh = []
    for det in pred:
        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:,:4], im0s.shape).round()
            for *xyxy, conf, _ in reversed(det):
                xywh.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())
    # print(xywh)
    points = pd.DataFrame(xywh, columns=['xcen', 'ycen', 'width', 'height'])
    points['xcen'] *= 1920
    points['ycen'] *= 1080
    points['width'] *= 1920
    points['height'] *= 1080
    # print(points)
    widths = np.array(points['width']) * 0.5 # width / 2
    heights = np.array(points['height']) * 0.5 # height / 2
    
    # get x0, y0, x1, y1
    x0_, y0_, x1_, y1_ = [], [], [], []
    for idx, p in points.iterrows():
        # print(idx, p)
        x0_.append(p['xcen'] - widths[idx])
        y0_.append(p['ycen'] - heights[idx])
        x1_.append(p['xcen'] + widths[idx])
        y1_.append(p['ycen'] + heights[idx])
    points['x0'] = x0_
    points['y0'] = y0_
    points['x1'] = x1_
    points['y1'] = y1_
    
    # draw_img = cv2.imread(path)
    # for idx, p in points.iterrows():
    #     draw_rectangle(draw_img, int(p['x0']), int(p['x1']), int(p['y0']), int(p['y1']), 'Green')
    # out.write(draw_img)
    # point_list = np.array(points.drop(['width', 'height'], axis=1))
    # for idx, i in enumerate(point_list):
    #     i[1] += heights[idx]
    # cnts = point_list.astype(int)
    # print('Final Contours')
    # print(cnts)
            
    # print('Detection Model Predition:')
    # print(pred)
    # print(f'Count = {len(pred[0])}')

# out.release()
print('Video Release !')
# command = 'python3 ./yolov5-master/detect.py --weights ./Models/cone_model_0705.pt --source \
#     /app/1028_frame/3.jpg --conf-thres 0.7 --save-txt --save-conf --nosave --project /app/results/modelvsdetect \
#     --exist-ok'.format(img_src)

model = torch.hub.load(('./yolov5-master'), 'custom', path = './Models/cone_model_0705.pt', source='local')
# model.model.warmup(imgsz=(1,3,640,640))
model.eval()
model.cuda()
model.conf = 0.7
# model.amp = True

img = cv2.imread('/app/1028_frame/3.jpg')
# data = LoadImages(img, img_size=imgsz, stride=stride, auto=pt)
# draw_img = img.copy()

# print(points)
# for idx, p in points.iterrows():
#     draw_rectangle(draw_img, int(p['x0']), int(p['x1']), int(p['y0']), int(p['y1']), 'Green')
# cv2.imwrite('/app/results/modelvsdetect/rec_detect.png', draw_img)
# print('Image Save !')

print(f'img type = {type(img)}')
# print(img.shape)
# print(f'test img type = {type(test_img)}')
# test_img = test_img.cpu().detach().numpy()
# test_img = np.squeeze(test_img).transpose(1, 2, 0)
# print(test_img.shape)
# result = model(img, augment=True)
result = model(im)
print(f'tensor result = {result}')
# print(f'tensor shape = {result.shape}')
# r_len = len(result.pandas().xywh[0])
# r_confs = np.array(result.pandas().xywh[0]['confidence']).tolist()
# print('model name = {}:'.format(model.__class__.__name__))
# print(f'count = {r_len}, conf = {r_confs}')
'''
# os.system(command)
# label_dir = '/app/results/modelvsdetect/exp/labels/3.txt'
# info = []
# with open(label_dir, 'r') as f:
#     info = f.readlines()
#     f.close()
# d_len = len(info)
# d_confs = []
# for line in info:
#     d_confs.append(float(line.split()[-1]))
# print('detect:')
# print(f'count = {d_len}, conf = {d_confs}')

from utils.augmentations import letterbox

# Image Pre-processing (Tensor)
def img_preprocessing(img_src, stride, pt):
    im = letterbox(img_src, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im) # contiguous
    im = torch.from_numpy(im).to(device)
    im = im.float()
    im /= 255 # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3: im = im[None] # expand for batch dim
    return im, img_src

img_src = cv2.imread('1028_frame/3.jpg')
device = torch.device('cuda')
data = 'yolov5-master/data/cone.yaml'
# weights = 'Models/cone_model_0705.pt'
# weights = 'Models/cone_model_0527.pt'
weights = 'Models/helmet_model_1028.pt'
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
# print(stride, names, pt)
imgsz = check_img_size(imgsz, s=stride)  # check image size
bs = 1  # batch_size
vid_path, vid_writer = [None] * bs, [None] * bs
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
model.eval()

im, im0 = img_preprocessing(img_src=img_src, stride=stride, pt=pt)
print('Image Process Done')

# Model Inference
def Inference(model, im, im0, conf_thres=0.6, classes=None, agnostic_nms=False, max_det=1000):
    # Predict
    pred = model(im, augment=False, visualize=False)
    # NMS
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45, classes=classes, agnostic=agnostic_nms, max_det=max_det)
    # Pandas Result
    xywh = []
    for det in pred:
        gn = torch.tensor(im0.shape)[[1,0,1,0]]
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:,:4], im0.shape).round()
            for *xyxy, conf, _ in reversed(det):
                xywh.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())
    points = pd.DataFrame(xywh, columns=['xcen', 'ycen', 'width', 'height'])
    # Get real-world position (CV2 image shape [HWC])
    print(im0.shape)
    points['xcen'] *= im0.shape[1]
    points['ycen'] *= im0.shape[0]
    points['width'] *= im0.shape[1]
    points['height'] *= im0.shape[0]
    # Get x0, y0, x1, y1
    widths = np.array(points['width']) * 0.5 # width / 2
    heights = np.array(points['height']) * 0.5 # height / 2
    x0_, y0_, x1_, y1_ = [], [], [], []
    for idx, p in points.iterrows():
        # x0_.append(p['xcen'] - widths[idx])
        # y0_.append(p['ycen'] - heights[idx])
        # x1_.append(p['xcen'] + widths[idx])
        # y1_.append(p['ycen'] + heights[idx])
        x0_.append(p['xcen'] - float(p['width'] * 0.5))
        y0_.append(p['ycen'] - float(p['height'] * 0.5))
        x1_.append(p['xcen'] + float(p['width'] * 0.5))
        y1_.append(p['ycen'] + float(p['height'] * 0.5))
    points['x0'] = x0_
    points['y0'] = y0_
    points['x1'] = x1_
    points['y1'] = y1_
    
    cnt_list = []
    for _, p in points.iterrows():
        cnt_list.append((p['xcen'], p['ycen'] + p['height'] * 0.5))
    cnt_list = np.array(cnt_list)
    cnts = cnt_list.astype(int)
    # print(cnts)
    re_cnts = np.reshape(cnts, (len(cnts), 2))
    # print(re_cnts)
    return points

draw_img = img_src.copy()

result = Inference(model, im, im0, classes=[0])
# print(result)
for idx, p in result.iterrows():
    draw_rectangle(draw_img, int(p['x0']), int(p['x1']), int(p['y0']), int(p['y1']), 'Green')

data='yolov5-master/data/coco.yaml'
weights = 'Models/yolov5s.pt'
human_model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
im, im0 = img_preprocessing(img_src=img_src, stride=human_model.stride, pt=human_model.pt)

h_result = Inference(human_model, im, im0, classes=[0])
for idx, p in h_result.iterrows():
    draw_rectangle(draw_img, int(p['x0']), int(p['x1']), int(p['y0']), int(p['y1']), 'Red')
cv2.imwrite('/app/results/modelvsdetect/helmet_test.png', draw_img)
print('Image Save !')