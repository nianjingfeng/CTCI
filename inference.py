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
import matplotlib.path as mpltPath
from datetime import datetime

'''YOLOv5 python packages'''
sys.path.insert(1, 'yolov5-master/')
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import Profile, check_file, check_img_size, non_max_suppression, scale_coords, xyxy2xywh
''''''

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Set current contours
current_contours = None
previous_contours = None
COUNTER = 300
cone_detect_counter = 0
MAX_LEN = 0
# Notification
CORDON = 0
Alert_MAX = 10

# Show notification message
def show_notification(person_id=None, coordinate=None, message=''):
    def print_eq():
        print('========================================')
    def print_mi():
        print('|                                      |')    
    message = message.center(38)
    id_message = None
    if person_id and coordinate:
        id_message = f'Person id \'{person_id}\' at X[{coordinate[0]}], Y[{coordinate[1]}]'
        id_message = id_message.center(38)
    print_eq()
    for i in range(2):
        print_mi()
    if id_message:
        print(f'|{id_message}|')
    else:
        print_mi()
    print(f'|{message}|')
    for i in range(2):
        print_mi()
    print_eq()

# Helmet Detection
def draw_helmet(helmet_model, src_img, output_img):
    helmet_result = helmet_model(src_img)
    helmet_points = helmet_result.pandas().xywh[0]
    # print('---------------------------------------------------')
    # print(f'helmet_count: {len(helmet_points)}')
    # print('---------------------------------------------------')
    x1s = [int(point['xcenter'] - (point['width'] / 2.0)) for idx, point in helmet_points.iterrows()]
    y1s = [int(point['ycenter'] - (point['height'] / 2.0)) for idx, point in helmet_points.iterrows()]
    x2s = [int(point['xcenter'] + (point['width'] / 2.0)) for idx, point in helmet_points.iterrows()]
    y2s = [int(point['ycenter'] + (point['height'] / 2.0)) for idx, point in helmet_points.iterrows()]
    # Draw bounging box
    for i in range(len(x1s)):
        draw_rectangle(output_img, 'helmet', x1s[i], x2s[i], y1s[i], y2s[i], color='Red')
    return output_img

# Human Prediction
def predict_human(human_model, src_img):
    human_result = human_model(src_img)
    # Show numbers of predicted human
    # print('---------------------------------------------------')
    # print(f'Human count: {len(human_result.pandas().xywh[0])}')
    # print('---------------------------------------------------')
    return human_result.pandas().xywh[0]

# VPS Prediction
def predict_vps(vps_model, src_img):
    vps_result = vps_model(src_img)
    # Show numbers of predicted vps
    # print('---------------------------------------------------')
    # print(f'VPS count: {len(vps_result.pandas().xywh[0])}')
    # print('---------------------------------------------------')
    return vps_result.pandas().xywh[0]

# Set Color List
def set_color(option=None):
    if option == 'Red':
        return (0, 0, 255)
    elif option == 'Blue':
        return (255, 0, 0)
    elif option == 'Green':
        return (0, 255, 0)
    else: return (0, 255, 0)

# Check detected cone number and positioin
def cone_similarity(cur_cone, det_cone):
    sim_thre = 10
    sim_count = 0
    for point in det_cone:
        for cur_point in cur_cone:
            # Euclidean Distance
            sim = math.sqrt(sum(pow(a-b,2) for a, b in zip(cur_point, point)))
            if sim <= sim_thre:
                sim_count += 1
                break
    if (sim_count < (len(cur_cone) // 2)): return False
    return True

# if people in danger zone without vps return true
def danger_zone_check(human_x1, human_x2, human_y1, human_y2, vps_tl, vps_br):
    check_flag = False
    d_thres = 30
    for i in range(len(vps_tl)):
        check_flag = check_flag or \
        (vps_tl[i][0] >= human_x1 and vps_tl[i][1] >= human_y1 and vps_br[i][0] <= human_x2 and vps_br[i][1] <= human_y2) \
        or ((abs(vps_tl[i][0] - human_x1) <= d_thres) \
        and (abs(vps_br[i][0] - human_x2) <= d_thres) \
        and (abs(vps_br[i][1] - human_y2) <= d_thres) \
        and (abs(vps_tl[i][1] - human_y1) <= d_thres))
    return not(check_flag)
    
def draw_rectangle(img, id, x1, x2, y1, y2, color): # top-left and bottom-right coordinates of rectangle
    box_label = 'id_' + id
    if id == 'vps':
        box_label = 'vps'
    elif id == 'helmet':
        box_label = 'helmet'
    color_list = (0, 0, 255)
    if color == 'Blue':
        color_list = (255, 0, 0)
    elif color == 'Red':
        color_list = (0, 0, 255)
        if id == 'helmet':
            color_list = (0, 127, 127)
    cv2.rectangle(img, (x1, y1), (x2, y2), color_list, 10)
    labelSize = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)
    _x1 = x1 # bottomleft x of text
    _y1 = y1 # bottomleft y of text
    _x2 = x1+labelSize[0][0] # topright x of text
    _y2 = y1-labelSize[0][1] # topright y of text
    cv2.rectangle(img, (_x1,_y1), (_x2,_y2), color_list, cv2.FILLED)
    cv2.putText(img, box_label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
    return

def cone_determine(cur_cone, new_cone):
    global MAX_LEN, COUNTER, cone_detect_counter
    if len(new_cone) > MAX_LEN: # max_len
        cone_detect_counter = 0
        MAX_LEN = len(new_cone)
        return new_cone
    elif cone_similarity(cur_cone, new_cone): # Occlude but same
        cone_detect_counter += 1
        if cone_detect_counter > COUNTER: # Re-detect
            if COUNTER == 50: COUNTER = 300 # Recovery
            cone_detect_counter = 0
            MAX_LEN = len(new_cone)
            return new_cone
        else: return cur_cone
    else: # Times up, update or cone_similarity not true
        cone_detect_counter += 1
        if cone_detect_counter > COUNTER: # Re-detect
            if COUNTER == 50: COUNTER = 300 # Recovery
            cone_detect_counter = 0
            MAX_LEN = len(new_cone)
            return new_cone
        elif not cone_similarity(cur_cone, new_cone):
            COUNTER = 50 # Update in 5s (quickly)
            print('Cone Similarity Not Same !')
            if cone_detect_counter > COUNTER:
                if COUNTER == 50: COUNTER = 300 # Recovery
                cone_detect_counter = 0
                MAX_LEN = len(new_cone)
                return new_cone
            else: return cur_cone

len_list = []

def cone_detection(model, img):
    cone_data = 'yolov5-master/data/cone.yaml'
    imgsz = (640, 640)
    dataset = LoadImages(img)
    cone_model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    pred = None
    for p, im, im0, v, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred)
    pred_ = pred[0].detach().cpu().numpy()
    conf = []
    for k in pred_:
        conf.append(k[4])
    xywh = []
    for det in pred:
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:,:4], im0.shape).round()
            for *xyxy, conf, _ in reversed(det):
                xywh.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())
    points = pd.DataFrame(xywh, columns=['xcen', 'ycen', 'width', 'height'])
    points['xcen'] *= 1920
    points['ycen'] *= 1080
    points['width'] *= 1920
    points['height'] *= 1080
    global len_list
    CONE_RATIO = 0.5
    # Get model predicted result
    # result = model(img, size=640) # print(), show(), save(), crop(), render(), pandas(), tolist(), __len__()
    # return pandas Dataframe from result
    # points = result.pandas().xywh[0] # xcenter, ycenter, width, height, confidence, class, name
    len_list.append(len(points))
    points = points.loc[(points['width'] > 50) & (points['height'] > 50)] # filter detected points bigger than setting (width and height > 50)
    if len(points) <= 2:
        return None
    # Get half of heigh of bounding box
    heights = np.array(points['height']) * CONE_RATIO
    # dropout unnecessary colums
    point_list = np.array(points.drop(['width', 'height'], axis=1))
    # point_list = np.array(points.drop(['width', 'height', 'confidence', 'class', 'name'], axis=1))
    # Get [xcenter, ydowncenter]
    for idx, i in enumerate(point_list):
        i[1] += heights[idx]
    cnts = point_list.astype(int)
    # Find convex points from point_list
    cnts = cv2.convexHull(cnts, returnPoints=True)
    re_cnts = np.reshape(cnts, (len(cnts), 2))
    return re_cnts

#process and upload the image
def Stream(idx, frame, cone_model, human_model, vps_model, helmet_model):
    # save_dir = os.path.join('results', 'test_new_vest', str(idx - 1306)) + '.jpg'
    save_dir = os.path.join('results', '1028_test', str(idx)) + '.jpg'
    with torch.no_grad():
        # Define global variable
        global CORDON, Alert_MAX, previous_contours, current_contours
        DATE_TIME = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        input_file = './image/'+DATE_TIME+'.jpg'
        color = 'Green' # ['Green', 'Red', 'Blue']
        # Set Convex Color, default Green
        color_list = set_color(color)
        # frame = cv2.imread(args.source)
        VPS_FLAG = True # VPS flag (True => Safe, False => Not Safe)
        # load cone image
        path = input_file
        pic_name = path.split('/')[-1]
        cone_image = frame
        # Cone detection and return contours, draw_img
        contours = cone_detection(cone_model, cone_image)
        cone_image = cv2.imread(frame)
        if contours is None:
            # Draw contours by points
            if current_contours is not None:
                frame = cv2.drawContours(cone_image, [current_contours], 0, color_list, 10)
            cv2.imwrite(save_dir, cone_image)
            return
        # current cone list determine
        if previous_contours is None and current_contours is None:
            previous_contours = np.copy(contours)
            current_contours = np.copy(contours)
            MAX_LEN = len(contours)
        else:
            current_contours = cone_determine(previous_contours, contours)
            previous_contours = np.copy(current_contours)

        # vps points
        VPS_RATIO = 0.5
        vps_points = predict_vps(vps_model, cone_image)
        v_widths = np.array(vps_points['width']) * VPS_RATIO
        v_height = np.array(vps_points['height']) * VPS_RATIO
        vps_np = np.array(vps_points.drop(['width', 'height', 'confidence', 'class', 'name'], axis=1))
        # vps_point top left coordinate
        vps_top_left = np.copy(vps_np)
        for idx, i in enumerate(vps_top_left):
            i[0] -= v_widths[idx]
            i[1] -= v_height[idx]
        vps_top_left = vps_top_left.astype(int)
        # vps_point bottom
        vps_bott_right = np.copy(vps_np)
        for idx, i in enumerate(vps_bott_right):
            i[0] += v_widths[idx]
            i[1] += v_height[idx]
        vps_bott_right = vps_bott_right.astype(int)

        # vps_point_for_show
        vps_bf = np.copy(vps_np)
        for idx, i in enumerate(vps_bf):
            i[0] -= v_widths[idx]
            i[1] += (v_height[idx] * 3 * 1.1)
        vps_bf = vps_bf.astype(int)

        vps_br = np.copy(vps_np)
        for idx, i in enumerate(vps_br):
            i[0] += v_widths[idx]
            i[1] += (v_height[idx] * 3 * 1.1)
        vps_br = vps_br.astype(int)

        # Set output image
        # Draw contours by points
        output_img = cv2.drawContours(cone_image, [current_contours], 0, color_list, 10)
        output_img = draw_helmet(helmet_model, cone_image, output_img)

        HUMAN_RATIO = 0.5
        # Check vps and non_vps in polygon
        poly_path = mpltPath.Path(current_contours) # Get cones polygon
        points = predict_human(human_model, cone_image) # Detected human points

        if len(points):
            # Create logfile using pandas
            log = pd.DataFrame(columns=['DateTime', 'X_pos', 'Y_pos', 'ID', 'SRC_PIC'])
            # Get width and height of each human bounding box
            widths = points['width']
            widths = np.array(widths) * HUMAN_RATIO
            heights = points['height']
            heights = np.array(heights) * HUMAN_RATIO
            # Dropout unnecessary colums
            points_ = points.drop(['width', 'height', 'confidence', 'class', 'name'], axis=1)
            np_points = np.array(points_)
            # Get [xcenter, y_downcenter]
            for idx, i in enumerate(np_points):
                i[1] += heights[idx]
            # Get bottom left of [x, y]
            human_point_left = np.copy(np_points)
            for idx, i in enumerate(human_point_left):
                i[0] -= widths[idx]
                if i[0] < 0: i[0] = 0
            human_point_left = human_point_left.astype(int)
            # Get bottom right of [x, y]
            human_point_right = np.copy(np_points)
            for idx, i in enumerate(human_point_right):
                i[0] += widths[idx]
            human_point_right = human_point_right.astype(int)
            # Detect persons that is in danger zone or not
            # Count for persons if in danger zone
            for i in range(len(np_points)):
                if poly_path.contains_points([human_point_left[i]]) or poly_path.contains_points([human_point_right[i]]):
                    # Person in danger zone
                    x1 = human_point_left[i][0]
                    x2 = human_point_right[i][0]
                    y1 = int(human_point_left[i][1] - 2 * heights[i])
                    y2 = human_point_right[i][1]
                    # Violation check
                    if danger_zone_check(x1, x2, y1, y2, vps_top_left, vps_bott_right):
                        draw_rectangle(output_img, str(i + 1), x1, x2, y1, y2, None) # Draw violation person
                        VPS_FLAG = False
                        # Logging
                        # new_log = {'DateTime':DATE_TIME, 'X_pos':np_points[i][0], 'Y_pos':np_points[i][1], 'ID':(i+1), 'SRC_PIC':pic_name}
                        # log = log.append(new_log, ignore_index=True)
            # Draw VPS in danger zone
            for i in range(len(vps_bf)):
                if poly_path.contains_points([vps_bf[i]]) or poly_path.contains_points([vps_br[i]]):
                    draw_rectangle(output_img, 'vps', vps_top_left[i][0], vps_bott_right[i][0], vps_top_left[i][1], vps_bott_right[i][1], 'Blue')
            # Notification (Continue in 10 Frames)
            if VPS_FLAG:
                CORDON += 1
                if CORDON >= Alert_MAX:
                    # show_notification(None, None, 'Notification')
                    # f_log_dir = os.path.join('.', 'results', 'log', save_dir.split('/')[-1])
                    # cv2.imwrite(f_log_dir, frame)
                    CORDON = 0
            else:
                CORDON = 0
            # save_dir = os.path.join('results', DATE_TIME) + '.jpg'
            cv2.imwrite(save_dir, output_img)
        else: cv2.imwrite(save_dir, output_img)

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT/'data/images', help='file path to detected')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    # cone_model = torch.hub.load(('./yolov5-master'), 'custom', path = './Models/cone_model_0705.pt', source='local')
    # cone_model.eval()
    # cone_model.cuda()
    # cone_model.conf = 0.7
    device = torch.device('cuda')
    
    cone_weights = 'Models/cone_model_0705.pt'
    cone_data = 'yolov5-master/data/cone.yaml'
    cone_model = DetectMultiBackend(cone_weights, device=device, data=cone_data)
    

    # Predict human
    human_model = torch.hub.load(('./yolov5-master'), 'custom', path = './Models/yolov5s.pt', source='local')
    human_model.eval()
    human_model.cuda()
    human_model.classes = [0] # person class in yolo
    human_model.conf = 0.65

    # Predict vps
    vps_model = torch.hub.load(('./yolov5-master'), 'custom', path = './Models/vps_model_1001.pt', source='local')
    vps_model.eval()
    vps_model.cuda()
    vps_model.conf = 0.7
    
    # Predict helmet
    helmet_model = torch.hub.load(('./yolov5-master'), 'custom', path = './Models/helmet_model_1028.pt', source='local')
    helmet_model.eval()
    helmet_model.cuda()
    helmet_model.classes = [0] # helmet class
    helmet_model.conf = 0.82
        
    args = parser_opt()
    # frames = glob.glob('1028_frame/*.jpg')
    # frames = trange(749, desc='Test')
    frames = range(0, 749)
    # frames = range(1306, 1756)
    for frame in tqdm(frames):
        # input_frame = cv2.imread(frame)
        # input_frame = cv2.imread(os.path.join('1028_frame', str(frame)) + '.jpg')
        input_frame = os.path.join('1028_frame', str(frame)) + '.jpg'
        # input_frame = cv2.imread(os.path.join('datasets/new_vest/images', str(frame)) + '_vest.jpg')
        # idx = int(frame.split('/')[-1][:-4])
        Stream(frame, input_frame, cone_model, human_model, vps_model, helmet_model)
    # with open('results/infe.txt', 'w') as f:
    #     for i in len_list:
    #         f.write(f'{i}\n')
    #     f.close() 
    # Stream(args, cone_model, human_model, vps_model, helmet_model)