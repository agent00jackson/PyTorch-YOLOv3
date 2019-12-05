from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import pandas as pd
import numpy as np

classes = ["person", "group"]
WeightsPath = "d:\\Datasets\\weights\\weights.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet("D:\\yolov3-custom.cfg", img_size=416).to(device)
model.load_state_dict(torch.load(WeightsPath))
model.eval()

def LoadImage(imgPath):
    # Extract image as PyTorch tensor
    img = transforms.ToTensor()(Image.open(imgPath))
    # Pad to square resolution
    img, _ = pad_to_square(img, 0)
    # Resize
    img = resize(img, 416)
    #Tensor = torch.cuda.FloatTensor
    return Variable(img.cuda())[None]

def extractFeatures(imgPath):
    img = LoadImage(imgPath)
    layer_outs = model(img)
    torch.cuda.empty_cache()
    return layer_outs[82]

res = {}
imgs = glob.glob("**/*.png")
for img in imgs:
    print(".", end="")
    ftrs = extractFeatures(img)
    name = os.path.basename(img)
    res[name] = ftrs

df = pd.DataFrame.from_dict(res)
df.to_csv("./results.txt", sep=",")