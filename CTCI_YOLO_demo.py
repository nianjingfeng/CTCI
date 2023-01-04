import torch
import cv2
import queue
import threading
import os
import math
import time
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.path as mpltPath
from datetime import datetime
from synology_api import filestation

# Set current contours
current_contours = None
COUNTER = 1200
cone_detect_counter = 0

#store the image camera collect
q = queue.LifoQueue()
#store the image after yolo processing
q_yolo = queue.LifoQueue()
#link to nas
nas_fs = filestation.FileStation('192.168.0.136', '5000',  "simslab", "Ntust11001013!", secure=False, cert_verify=False, dsm_version=7, debug=True, otp_code=None)
nas_path = "/surveillance/Generic_ONVIF-001/test/"
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Helmet Detection
def draw_helmet(helmet_model, src_img, output_img):
    helmet_result = helmet_model(src_img, size=1080)
    helmet_points = helmet_result.pandas().xywh[0]
    print('---------------------------------------------------')
    print(f'helmet_count: {len(helmet_points)}')
    print('---------------------------------------------------')
    x1s = [int(point['xcenter'] - (point['width'] / 2.0)) for idx, point in helmet_points.iterrows()]
    y1s = [int(point['ycenter'] - (point['height'] / 2.0)) for idx, point in helmet_points.iterrows()]
    x2s = [int(point['xcenter'] + (point['width'] / 2.0)) for idx, point in helmet_points.iterrows()]
    y2s = [int(point['ycenter'] + (point['height'] / 2.0)) for idx, point in helmet_points.iterrows()]
    # Draw bounging box
    for i in range(len(x1s)):
        draw_rectangle(output_img, 'helmet', x1s[i], x2s[i], y1s[i], y2s[i], color='Red')
    return output_img

def cone_similarity(cur_cone, det_cone):
    sim_thre = 30
    sim_flag = False
    for point in det_cone:
        for cur_point in cur_cone:
            # Euclidean Distance
            sim = math.sqrt(sum(pow(a-b,2) for a, b in zip(cur_point, point)))
            if sim <= sim_thre:
                sim_flag = True
                break
            else: sim_flag = False
    if len(cur_cone) < len(det_cone) and sim_flag: # When cone detection detect new cone at same scence
        return False
    return sim_flag

# if people in danger zone without vps return true
def danger_zone_check(human_x1, human_x2, human_y1, human_y2, vps_tl, vps_br):
    check_flag = False
    for i in range(len(vps_tl)):
        check_flag = check_flag or (vps_tl[i][0] >= human_x1 and vps_tl[i][1] >= human_y1 and vps_br[i][0] <= human_x2 and vps_br[i][1] <= human_y2)
    return not(check_flag)
    
def draw_rectangle(img, id, x1, x2, y1, y2, color): # top-left and bottom-right coordinates of rectangle
    box_label = 'id_' + id
    if id == 'vps':
        box_label = 'vps'
    # print(x1, x2, y1, y2)
    color_list = (0, 0, 255)
    if color == 'Blue':
        color_list = (255, 0, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color_list, 10)
    labelSize = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)
    _x1 = x1 # bottomleft x of text
    _y1 = y1 # bottomleft y of text
    _x2 = x1+labelSize[0][0] # topright x of text
    _y2 = y1-labelSize[0][1] # topright y of text
    cv2.rectangle(img, (_x1,_y1), (_x2,_y2), color_list, cv2.FILLED)
    cv2.putText(img, box_label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
    return

def cone_detection(model, img, conf_thres):
    CONE_RATIO = 0.5
    # Set model's output confidence
    model.conf = conf_thres
    print(f'model confidence threshold: {conf_thres}')
    # Get model predicted result
    result = model(img) # print(), show(), save(), crop(), render(), pandas(), tolist(), __len__()
    # return pandas Dataframe from result
    points = result.pandas().xywh[0] # xcenter, ycenter, width, height, confidence, class, name
    if len(points) > 0:
        print('---------------------------------------------------')
        print('Each detected object\'s confidence score:')
        print(points['confidence'])
        print('---------------------------------------------------')
        if len(points) <= 2:
            print('There is no polygon zone for further detection !')
            print('---------------------------------------------------')
            return 2, None
    else:
        print('---------------------------------------------------')
        print('Cone_model did not predict anything !')
        print('---------------------------------------------------')
        return None, None
    # Get half of heigh of bounding box
    heights = points['height']
    heights = np.array(heights) * CONE_RATIO
    # dropout unnecessary colums
    points = points.drop(['width', 'height', 'confidence', 'class', 'name'], axis=1)
    # Get center point as a list
    point_list = np.array(points)
    if len(point_list) > 0:
        # Get [xcenter, ydowncenter]
        for idx, i in enumerate(point_list):
            i[1] += heights[idx]
        cnts = point_list.astype(int)
        # Find convex points from point_list
        cnts = cv2.convexHull(cnts, returnPoints=True)
        re_cnts = np.reshape(cnts, (len(cnts), 2))
        if len(cnts) > 0:
            return re_cnts, img
        else:
            print('Cone_model did not predict anything !')
            return None, None

#collect the image
def Receive():
    URL = "rtsp://syno:91c7179d596a37c0260aa3abad7ef55e@192.168.0.136:554/Sms=7.unicast"
    ipcam = cv2.VideoCapture(URL)
    success, frame = ipcam.read()
    q.put(frame)
    while success:
        success, frame = ipcam.read()
        q.put(frame)


#process and upload the image
def Stream(cone_model, human_model, vps_model, helmet_model):
    while True:
        if q.empty() != True:
            # Danger Zone Detection (YOLOv5)
            frame = q.get()
            global cone_detect_counter, current_contours
            DATE_TIME = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            input_file = './image/'+DATE_TIME+'.jpg'
            conf_thres = 0.6 # Cone Detection Confidence Threshold
            color = 'Green' # ['Green', 'Red', 'Blue']
            cone_detect_counter += 1

            VPS_FLAG = True # VPS flag (True => Safe, False => Not Safe)
            
            # load cone image
            path = input_file
            pic_name = path.split('/')[-1]
            cv2.imwrite('./image/'+DATE_TIME+'.jpg',frame)
            cone_image = frame
            # Cone detection and return contours, draw_img
            contours, cone_img = cone_detection(cone_model, cone_image, conf_thres)
            if contours is None and cone_img is None:
                print('Detection Error: None of cone detected !')
                continue
            elif contours is not None and cone_img is None:
                print('Further detection failed: No enough cones for generating a polygon zone !')
                continue
            
            # Replace new cone contours list
            if current_contours is None and contours is not None:
                current_contours = contours # Cone contours list
            elif cone_detect_counter >= COUNTER or len(current_contours) != len(contours):
                # cone contours similarity
                if not cone_similarity(current_contours, contours):
                    print('---------------------------------------------------')
                    print('New Cone Contours Detected ! Replace Old One ...')
                    print('---------------------------------------------------')
                    current_contours = contours
                if cone_detect_counter >= COUNTER:
                    cone_detect_counter = 0
                    print('Cone Counter Reset ...')
            else:
                print('---------------------------------------------------')
                print('Using Current Cones for danger zone !')
                print('---------------------------------------------------')

            human_result = human_model(cone_image)
            # Show numbers of predicted human
            print('---------------------------------------------------')
            print(f'Human count: {len(human_result.pandas().xywh[0])}')
            print('---------------------------------------------------')
            person_img = cone_image

            vps_result = vps_model(person_img)
            # Show numbers of predicted vps
            print('---------------------------------------------------')
            print(f'VPS count: {len(vps_result.pandas().xywh[0])}')
            print('---------------------------------------------------')

            # vps points
            VPS_RATIO = 0.5
            vps_points = vps_result.pandas().xywh[0]
            v_widths = np.array(vps_points['width']) * VPS_RATIO
            v_height = np.array(vps_points['height']) * VPS_RATIO
            vps_np = np.array(vps_points.drop(['width', 'height', 'confidence', 'class', 'name'], axis=1))
            vps_center = np.copy(vps_np)
            vps_center = vps_center.astype(int)
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

            # Set output image
            # Set Convex Color, default Green
            color_list = (0, 255, 0)
            if color == 'Red':
                color_list = (0, 0, 255)
            elif color == 'Blue':
                color_list = (255, 0, 0)
            # Draw contours by points
            output_img = cv2.drawContours(cone_image, [current_contours], 0, color_list, 10)
            output_img = draw_helmet(helmet_model, cone_image, output_img)

            HUMAN_RATIO = 0.5
            # Check vps and non_vps in polygon
            print(f'contours = {current_contours}')
            print('---------------------------------------------------')
            poly_path = mpltPath.Path(current_contours) # Get cones polygon
            points = human_result.pandas().xywh[0] # Detected human points

            if len(points) > 0:
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
                person_count = 0 # Count for person in danger zone
                # Count for persons if in danger zone
                for i in range(len(np_points)):
                    if poly_path.contains_points([human_point_left[i]]) or poly_path.contains_points([human_point_right[i]]):
                        # Person in danger zone, draw rectangle
                        x1 = human_point_left[i][0]
                        x2 = human_point_right[i][0]
                        y1 = int(human_point_left[i][1] - 2 * heights[i])
                        y2 = human_point_right[i][1]
                        # Person id start from 1
                        if danger_zone_check(x1, x2, y1, y2, vps_top_left, vps_bott_right):
                            VPS_FLAG = False
                            draw_rectangle(output_img, str(i + 1), x1, x2, y1, y2, None)
                            print(f'Person id {i + 1} did not have a VPS at coordinate {np_points[i]} in dangerzone')
                            new_log = {'DateTime':DATE_TIME, 'X_pos':np_points[i][0], 'Y_pos':np_points[i][1], 'ID':(i+1), 'SRC_PIC':pic_name}
                            log = log.append(new_log, ignore_index=True)
                        else:
                            draw_rectangle(output_img, 'vps', x1, x2, y1, y2, 'Blue')
                        person_count += 1
                print(f'Person count: {person_count}')
                if person_count > len(vps_result.pandas().xywh[0]):
                    VPS_FLAG = False
                print('---------------------------------------------------')
                if not(VPS_FLAG):
                    print('Notification: There are some violators in danger zone !!!')
                else:
                    print('Safe Now !')
                print('---------------------------------------------------')
            #nas_fs.upload_file(dest_path=nas_path+time_date+"/"+time_hour, file_path='./cone/'+time_now+".JPG", create_parents=True, overwrite=False, verify=False)
            
            # cv2.imwrite('./cone/'+DATE_TIME+'.jpg',output_img)
            q_yolo.put(output_img)
#show the image after processing
def Display():
    
    while True:
        if q_yolo.empty() != True:
            frame_yolo = q_yolo.get()
            try:
                cv2.imshow("Stream",frame_yolo)
            except:
                pass
            q.queue.clear()
            q_yolo.queue.clear()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    cone_model = torch.hub.load(('./yolov5-master'), 'custom', path = './Models/cone_model_0527.pt', source='local')
    cone_model.eval()
    cone_model.cuda()

    # Predict human
    human_model = torch.hub.load(('./yolov5-master'), 'custom', path = './Models/yolov5s.pt', source='local')
    human_model.eval()
    human_model.cuda()
    human_model.classes = [0] # person class in yolo
    human_model.conf = 0.65

    # Predict vps
    vps_model = torch.hub.load(('./yolov5-master'), 'custom', path = './Models/vps_model.pt', source='local')
    vps_model.eval()
    vps_model.cuda()
    vps_model.conf = 0.7
    
    # Predict helmet
    helmet_model = torch.hub.load(('./yolov5-master'), 'custom', path = './Models/helmet_model_1026.pt', source='local')
    helmet_model.eval()
    helmet_model.cuda()
    helmet_model.classes = [1] # helmet class
    helmet_model.conf = 0.7

    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Stream, args=(cone_model, human_model, vps_model))
    p3 = threading.Thread(target=Display)   
    p1.start()
    p2.start()
    p3.start()