from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

WEIGHTS = "d:\\Datasets\\weights\\weights.pth"
CLASSES = ["Person", "Group"]
VIDEO = "d:\\Datasets\\Caltech\\atlDrive.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet("D:\\yolov3-custom.cfg", img_size=416).to(device)
model.load_state_dict(torch.load(WEIGHTS))
model.eval()

vid = cv2.VideoCapture(VIDEO)
with torch.no_grad():
    lastTime = time.time()
    while(vid.isOpened()):
        ret, frame = vid.read()
        if not ret:
            continue
        resized = cv2.resize(frame.copy(), (416, 416))
        imgArray = np.array([resized])
        imgArray = np.rollaxis(imgArray, 3, 1)
        imgTensor = torch.from_numpy(imgArray).cuda().float()

        detections = model(imgTensor)
        detections = non_max_suppression(detections, .5, .4)
        
        if detections[0] is not None:
            detections = detections[0]
            try:
                detections = rescale_boxes(detections, 416, frame.shape[:2])
            except ValueError:
                pass

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                y1 = -1 * y1
                y2 = -1 * y2

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (147, 20, 255), 3)
        newTime = time.time()
        deltaTime = newTime - lastTime
        lastTime = newTime
        fps = 1/deltaTime
        frame = cv2.putText(frame, str(int(round(fps))), (10,10), cv2.FONT_HERSHEY_SIMPLEX, .5, (147, 20, 255), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
          break
