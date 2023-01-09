# CTCI

## Guide

This project aims to detect the unqualified personnel in the dangerous zone of the construction site. This model applies object detecion technology YOLOv5 to detect the traffic cone, human and safety equipment. 
<br>[Click here](https://github.com/ultralytics/yolov5) to know more information about YOLOv5.

Setting
---
There are 4 equipments of this project:<br>
1. Synology NAS
2. IP camera
3. Server
4. Arduino beeper

We use Synology Surveillance system to link the ipcam, the system provide the url thus you can get the image by the system. If model detects the unqualified personnel, the beeper will announce and the image will be uploaded to the NAS.

Models
---
There are 4 models to inference:
1. Traffic Cone: Detect the traffic cone and fence up the dangerous zone.
2. Human: We use pretrain model from coco dataset to detect the human. The model will check the relative position of the vest, helmet and the human body to judge whether it is worn correctly.
3. Vest: To recognize the qualified personnel, the model will detect the vest first.
4. Helmet: After detecting the vest, the model will detect the helmet. If one them is not detected, the system will alarm.

Inference
---
[Inference Video](https://drive.google.com/file/d/1tl-t3xN8XDajyEb5_lnlssTZN5M1SrUd/view?usp=share_link)