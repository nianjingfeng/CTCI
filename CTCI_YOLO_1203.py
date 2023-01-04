'''========================================================'''
''' PYTHON PACKAGES '''
'''========================================================'''
import os
import cv2
import sys
import math
import torch
import queue
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.path as mpltPath
from synology_api import filestation
import serial
'''========================================================'''
''' YOLOv5 PACKAGES '''
'''========================================================'''
sys.path.insert(1, 'yolov5-master/') # Insert python import path
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
'''========================================================'''
''' NAS CONNECTION '''
'''========================================================'''
#store the image camera collect
q = queue.LifoQueue()
#store the image after yolo processing
q_yolo = queue.LifoQueue()
#link to nas
nas_fs = filestation.FileStation('192.168.0.136', '5000',  "simslab", "Ntust11001013!", secure=False, cert_verify=False, dsm_version=7, debug=True, otp_code=None)
nas_path = "/surveillance/Generic_ONVIF-001/test/"
os.environ['MKL_THREADING_LAYER'] = 'GNU'
'''========================================================'''
''' GLOBAL VARIABLES '''
'''========================================================'''
# Models
CONE_MODEL = 'Models/cone_model_0705.pt'
HUMAN_MODEL = 'Models/yolov5s.pt'
VPS_MODEL = 'Models/vps_model_1001.pt'
HELMET_MODEL = 'Models/helmet_model_1218.pt'
# Data
CONE_DATA = 'yolov5-master/data/cone.yaml'
HUMAN_DATA = 'yolov5-master/data/coco.yaml'
VPS_DATA = 'yolov5-master/data/vest.yaml'
HELMET_DATA = 'yolov5-master/data/helmet.yaml'
'''========================================================'''
''' TORCH SETTINGS '''
'''========================================================'''
DEVICE = torch.device('cuda') # Set for GPU
'''========================================================'''
''' STATUS '''
'''========================================================'''
# Current Status
class C_Status:
    current_contours = None
    human_list = pd.DataFrame()
    violate_list = pd.DataFrame()
    qualified_list = pd.DataFrame()
    MAX_LEN = 0
    Reset_counter = 0
    V_FLAG = False # Violation Flag (Flase => No Violation, True => Violation)
    Continue_S = 0 # Use for continuing violation
    Img_src = None
    Img_draw = None
    def __init__(self):
        super().__init__()
    def setImg(self, img):
        self.Img_src = img.copy() # Load Image
        self.Img_draw = img # Image for Draw
    def setFlag(self, bool):
        self.V_FLAG = bool
    def similarity(self, contours):
        sim_thres = 10
        count = 0
        for p in contours:
            for cur_p in self.current_contours:
                sim_score = math.sqrt(sum(pow(a-b,2) for a, b in zip(p, cur_p))) # Euclidean Distance
                if sim_score <= sim_thres:
                    count += 1
                    break
        return True if count >= math.ceil(len(contours) / 2) else False
    def setContour(self, contours):
        if self.current_contours is None:
            self.current_contours = contours
            self.MAX_LEN = len(self.current_contours)
            self.Reset_counter = 0
        else:
            if len(contours) > self.MAX_LEN: # Cond 1. Max len
                self.current_contours = contours
                self.MAX_LEN = len(contours)
                self.Reset_counter = 0
                return
            self.Reset_counter += 1
            if self.similarity(contours):
                if self.Reset_counter > 300: # Re-detect
                    self.current_contours = contours
                    self.MAX_LEN = len(contours)
                    self.Reset_counter = 0
            else: # Time's up, update or similarity not true
                if self.Reset_counter > 50: # Re-detect in 5s (quickly)
                    self.current_contours = contours
                    self.MAX_LEN = len(contours)
                    self.Reset_counter = 0
    def get_Img(self, select=0):
        # 0 for source image, 1 for draw image
        if select==0:
            return self.Img_src
        elif select==1:
            return self.Img_draw 
        else: raise Exception('Image ERROR: No Image Load')
    def get_Contours(self):
        return self.current_contours
    def get_CStatus(self):
        return self.Continue_S
    def drawImg(self, x0, y0, x1, y1, color='green', box_label:str=None):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        _C = color.lower()
        COLOR_ = None
        if _C == 'green': COLOR_ = (0, 255, 0)
        elif _C == 'blue': COLOR_ = (255, 0, 0)
        elif _C == 'red': COLOR_ = (0, 0, 255)
        else: COLOR_ = (0, 127, 127)
        cv2.rectangle(self.Img_draw, (x0, y0), (x1, y1), COLOR_, 10)
        if box_label:
            labelSize = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)
            _x0 = x0 # bottomleft x of text
            _y0 = y0 # bottomleft y of text
            _x1 = x0 + labelSize[0][0] # topright x of text
            _y1 = y0 - labelSize[0][1] # topright y of text
            cv2.rectangle(self.Img_draw, (_x0,_y0), (_x1,_y1), COLOR_, cv2.FILLED)
            cv2.putText(self.Img_draw, box_label, (x0, y0), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,0), 1, cv2.LINE_AA)
    def draw_by_self(self, mode=0):
        if self.current_contours is not None:
            self.Img_draw = cv2.drawContours(self.Img_draw, [self.current_contours], 0, (0, 255, 0), 10) # Draw Cone with Green Color
        if mode:
            return self.Img_draw
    def insert_point(self, p, mode=0, u=0):
        # 0 for human in contours, 1 for realy violate human in contours, 2 for previous qualified human point
        if mode == 0:
            self.human_list = pd.concat([self.human_list, p], ignore_index=True).astype(int)
        elif mode == 1:
            self.violate_list = pd.concat([self.violate_list, p], ignore_index=True).astype(int)
        elif mode == 2:
            # Init qualify point
            p = pd.DataFrame([p.tolist()], columns=['xcen', 'ycen', 'width', 'height', 'x0', 'y0', 'x1', 'y1'])
            p['update'] = u
            self.qualified_list = pd.concat([self.qualified_list, p], ignore_index=True).astype(int)
    def reset_point(self, mode=0):
        # 0 for reset uncertain human, 1 for reset violate human
        if mode == 0:
            self.human_list = pd.DataFrame()
        elif mode == 1:
            self.violate_list = pd.DataFrame()
    def get_human(self, mode=0):
        # 0 for uncertain human, 1 for violate human
        if mode == 0:
            return self.human_list if self.human_list is not None else None
        elif mode == 1:
            return self.violate_list if self.violate_list is not None else None
    def reset_status(self):
        self.human_list = pd.DataFrame()
        self.violate_list = pd.DataFrame()
        self.Img_src = None
        self.Img_draw = None
        self.V_FLAG = False
C_S = C_Status() # Set up a new status
'''========================================================'''
#collect the image
def Receive():
    URL = "rtsp://syno:91c7179d596a37c0260aa3abad7ef55e@192.168.0.136:554/Sms=7.unicast"
    ipcam = cv2.VideoCapture(URL)
    success, frame = ipcam.read()
    q.put(frame)
    while success:
        success, frame = ipcam.read()
        q.put(frame)

#show the image after processing
def Display():
    out = cv2.VideoWriter('/home/will/Desktop/CTCI/CTCI-main/1220.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (1920, 1080))
    while True:
        if q_yolo.empty() != True:
            frame_yolo = q_yolo.get()
            try:
                cv2.imshow("Stream", frame_yolo)
                out.write(frame_yolo)
            except:
                pass
            q.queue.clear()
            q_yolo.queue.clear()
        if cv2.waitKey(1) == ord('q'):
            out.release()
            cv2.destroyAllWindows()
            break

# Load yolo model
def Load_YOLO_Model(weight, data):
    model = DetectMultiBackend(weights=weight, device=DEVICE, data=data)
    imgsz = (640, 640)
    model.warmup(imgsz=(1,3, *imgsz))
    model.eval()
    return model

# Image Pre-processing (Tensor)
def img_preprocessing(img_src, stride, pt):
    im = letterbox(img_src, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im) # contiguous
    im = torch.from_numpy(im).to(DEVICE)
    im = im.float()
    im /= 255 # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3: im = im[None] # expand for batch dim
    return im, img_src

# Model Inference (Return DataFrame Results)
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
    points['xcen'] *= im0.shape[1]
    points['ycen'] *= im0.shape[0]
    points['width'] *= im0.shape[1]
    points['height'] *= im0.shape[0]
    # Get x0, y0, x1, y1
    widths = np.array(points['width']) * 0.5 # width / 2
    heights = np.array(points['height']) * 0.5 # height / 2
    x0_, y0_, x1_, y1_ = [], [], [], []
    for idx, p in points.iterrows():
        x0_.append(p['xcen'] - widths[idx])
        y0_.append(p['ycen'] - heights[idx])
        x1_.append(p['xcen'] + widths[idx])
        y1_.append(p['ycen'] + heights[idx])
    points['x0'] = x0_
    points['y0'] = y0_
    points['x1'] = x1_
    points['y1'] = y1_
    return points

# Get Overlap Ratio from two bbox (intersection / obj_b)
def get_iou(bbox_human, bbox_obj):
    # x, y, w, h
    iou_x = max(bbox_human['x0'], bbox_obj['x0']) # x
    iou_y = max(bbox_human['y0'], bbox_obj['y0']) # y
    iou_w = min(bbox_human['x1'], bbox_obj['x1']) - iou_x # w
    iou_w = max(iou_w, 0)
    iou_h = min(bbox_human['y1'], bbox_obj['y1']) - iou_y # h
    iou_h = max(iou_h, 0)
    iou_area = iou_w * iou_h
    obj_area = bbox_obj['width'] * bbox_obj['height']
    return max(iou_area/obj_area, 0)

# Cone Detection (Detect & Draw Contours)
def Cone_Detection(cur_status: C_Status, cone_model):
    # Predict
    img_src = cur_status.get_Img(select=0)
    im, im0 = img_preprocessing(img_src=img_src, stride=cone_model.stride, pt=cone_model.pt)
    cone_results = Inference(model=cone_model, im=im, im0=im0, conf_thres=0.7)
    if len(cone_results.index) == 0: return False
    # Get Contours (Down Center Points)
    cnt_list = []
    for _, p in cone_results.iterrows():
        cnt_list.append((p['xcen'], p['ycen'] + float(p['height']) * 0.5))
    cnt_list = np.array(cnt_list)
    cnts = cnt_list.astype(int)
    # Find convex points from cnt_list
    cnts = cv2.convexHull(cnts, returnPoints=True)
    cnts = np.reshape(cnts, (len(cnts), 2))
    if len(cnts) > 2:
        # Cone Determine
        cur_status.setContour(cnts)
    cur_status.draw_by_self()
    return True

# Human Detection (Detect in Contours)
def Human_Detection(cur_status: C_Status, Poly, human_model):
    # Predict
    img_src = cur_status.get_Img(select=0)
    im, im0, = img_preprocessing(img_src=img_src, stride=human_model.stride, pt=human_model.pt)
    human_results = Inference(model=human_model, im=im, im0=im0, classes=[0], conf_thres=0.7)
    if len(human_results.index) > 0:
        # Get down_cen_right & down_cen_left
        dc_right, dc_left = [], []
        for _, p in human_results.iterrows():
            dc_right.append((p['x1'] - p['width'], p['y1'] + 5))
            dc_left.append((p['x1'], p['y1'] + 5))
        dc_right = np.array(dc_right).astype(int)
        dc_left = np.array(dc_left).astype(int)
        Cover_Flag = False
        for i in range(len(human_results)):
            if Poly.contains_points([dc_right[i]]) or Poly.contains_points([dc_left[i]]):
                # Person in Contours
                Cover_Flag = True
                cur_status.insert_point(human_results.iloc[[i]], mode=0)
        return Cover_Flag
    else: return False

# VPS Detection (Detect & Check)
def VPS_Detection(cur_status: C_Status, vps_model):
    # Predict
    img_src = cur_status.get_Img(select=0)
    im, im0, = img_preprocessing(img_src=img_src, stride=vps_model.stride, pt=vps_model.pt)
    vps_results = Inference(model=vps_model, im=im, im0=im0, conf_thres=0.7)
    # Check VPS for each human in Contours
    humen = cur_status.get_human(mode=0)
    cur_status.reset_point(mode=0)
    if len(vps_results.index) == 0 and len(humen.index) > 0:
        cur_status.setFlag(True)
        cur_status.violate_list = pd.concat([cur_status.violate_list, humen], ignore_index=True).astype(int)
        return
    elif len(vps_results) < len(humen): cur_status.setFlag(True) # Human more than VPS in Contours
    for idx, human_p in humen.iterrows():
        inside_Flag = False
        for _, vps_p in vps_results.iterrows():
            # Overlap threshold
            if get_iou(human_p, vps_p) > 0.8:
                inside_Flag = True
                break
        if inside_Flag: cur_status.insert_point(humen.iloc[[idx]], mode=0) # Not Violate Yes
        else:
            cur_status.insert_point(humen.iloc[[idx]], mode=1) # Already violate
            cur_status.setFlag(True)

# Helmet Detection (Detect & Check)
def Helmet_Detection(cur_status: C_Status, helmet_model):
    # Predict
    img_src = cur_status.get_Img(select=0)
    im, im0, = img_preprocessing(img_src=img_src, stride=helmet_model.stride, pt=helmet_model.pt)
    helmet_results = Inference(model=helmet_model, im=im, im0=im0, classes=[0], conf_thres=0.82)
    # Check helmet for each human in contours
    humen = cur_status.get_human(mode=0)
    if len(helmet_results.index) == 0 and len(humen.index) > 0:
        cur_status.setFlag(True)
        cur_status.violate_list = pd.concat([cur_status.violate_list, cur_status.human_list], ignore_index=True).astype(int)
        return
    elif len(helmet_results) < len(humen): cur_status.setFlag(True) # Human more than helmet in Contours
    for idx, human_p in humen.iterrows():
        inside_Flag = False
        for _, helmet_p in helmet_results.iterrows():
            # Overlap threshold
            if get_iou(human_p, helmet_p) > 0.7 and (human_p['y0'] <= helmet_p['ycen'] <= ((human_p['y0'] + human_p['ycen']) / 2)):
                inside_Flag = True
                break
        if inside_Flag:
            cur_status.insert_point(humen.iloc[[idx]], mode=0, u=0)
            C_S.drawImg(human_p['x0'], human_p['y0'], human_p['x1'], human_p['y1'], color='any', box_label=None)
        else:
            cur_status.insert_point(humen.iloc[[idx]], mode=1) # Violate human
            cur_status.setFlag(True)

# Evaluate the violate list point is qualified before and tracking qualified human
def Qualify(cur_status: C_Status):
    results = [True for i in range(len(cur_status.violate_list.index))]
    # Qualified list init
    if len(cur_status.qualified_list.index) == 0:
        # Insert point into qualified list
        for _, p in cur_status.human_list.iterrows():
            cur_status.insert_point(p, mode=2, u=0)
        return results
    else:
        remove_list = []
        # Check humen in violate list is qualified before and update bbox
        for v_idx, vio_p in cur_status.violate_list.iterrows():
            for q_idx, qua_p in cur_status.qualified_list.iterrows():
                if get_iou(vio_p, qua_p) > 0.85: # Same human from two adjacent frames (t, t+1), maybe not qualified anymore
                    u_tmp = qua_p['update'] + 1
                    cur_status.insert_point(vio_p, mode=2, u=u_tmp)
                    remove_list.append(q_idx)
                    # results[v_idx] = False
                    break
        for idx, qua_p in cur_status.qualified_list.iterrows():
            if idx not in remove_list:
                qua_p['update'] += 2
            if qua_p['update'] > 20:
                remove_list.append(idx)
        cur_status.qualified_list = cur_status.qualified_list.drop(cur_status.qualified_list.index[remove_list]).reset_index(drop=True)
        return results

#process and upload the image
def Stream(cone_model, human_model, vps_model, helmet_model):
    with torch.no_grad():
        while True:
            ser=serial.Serial("/dev/ttyUSB0",9600,bytesize=8,stopbits=1) 
            if q.empty() != True:
                # DATE_TIME = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) # Get Current Date Time
                Image_frame = q.get() # Get Image Frame from Camera as type<numpy array>
                C_S.setImg(Image_frame) # Load Image
                Step_1 = Cone_Detection(C_S, cone_model=cone_model)
                if not Step_1: # No polygon detected, next Image
                    q_yolo.put(C_S.draw_by_self(mode=1))
                    continue
                poly_path = mpltPath.Path(C_S.get_Contours())
                Step_2 = Human_Detection(C_S, poly_path, human_model=human_model)
                if not Step_2: # No human detected or not in polygon
                    q_yolo.put(C_S.draw_by_self(mode=1))
                    continue
                VPS_Detection(C_S, vps_model=vps_model)
                if C_S.get_human(mode=0) is not None:
                    Helmet_Detection(C_S, helmet_model=helmet_model)
                # Conditional continuous counting violation time
                if C_S.V_FLAG: C_S.Continue_S += 1
                else: C_S.Continue_S = 0
                # Get qualified human points if exist
                qua_list = Qualify(C_S)
                if C_S.get_CStatus() > 20:
                    # Draw violate human at output image
                    violate_list = C_S.get_human(mode=1)
                    if violate_list is not None and len(qua_list) > 0:
                        for v_idx, p in violate_list.iterrows():
                            if qua_list[v_idx]:
                                C_S.drawImg(p['x0'], p['y0'], p['x1'], p['y1'], color='red', box_label=str(v_idx))
                        # Notification (Output suspicious image to local and alert)
                        if sum(qua_list):
                            ser.write(chr(0X11).encode('utf-8'))
                            ser.write(chr(0X18).encode('utf-8'))
                ser.write(chr(0X28).encode('utf-8'))
                ser.write(chr(0X21).encode('utf-8'))
                q_yolo.put(C_S.get_Img(select=1))
                C_S.reset_status()
                    
# Main Function (Enter Point)
if __name__ == '__main__':
    # Load Model
    cone_model = Load_YOLO_Model(weight=CONE_MODEL, data=CONE_DATA)
    human_model = Load_YOLO_Model(weight=HUMAN_MODEL, data=HUMAN_DATA)
    vps_model = Load_YOLO_Model(weight=VPS_MODEL, data=VPS_DATA)
    helmet_model = Load_YOLO_Model(weight=HELMET_MODEL, data=HELMET_DATA)
    # Models = [cone_model, human_model, vps_model, helmet_model]
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Stream, args=(cone_model, human_model, vps_model, helmet_model))
    p3 = threading.Thread(target=Display)   
    p1.start()
    p2.start()
    p3.start()