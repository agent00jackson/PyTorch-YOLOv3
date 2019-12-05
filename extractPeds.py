import cv2
import numpy as np
import os
from multiprocessing import Pool
import glob

calWidth = 640
calHeight = 480

def getLines(filePath):
    lines = []
    with open(filePath) as f:
        for line in f:
            line.strip()
            lines.append(line)
    return lines

def resolveLabels(imgPath):
    name = os.path.basename(imgPath)
    name.replace("set", "train")
    name.replace(".png", ".txt")
    return "D:\\Datasets\\Caltech\\conv\\labels\\" + name 

def convBoxCoord(entry):
    words = entry.split(" ")
    x0 = (float(words[1]) * calWidth) + (calWidth / 2)
    y0 = (float(words[2]) * calHeight) + (calHeight / 2)
    width = float(words[3]) * calWidth
    height = float(words[4]) * calHeight
    x1 = x0 + width
    y1 = y0 + height
    if words[0] == "0":
        ret = True
    return (ret, round(x0), round(y0), round(x1), round(y1))

def getLabels(imgPath):
    labelPath = resolveLabels(imgPath)
    entries = getLines(labelPath)
    coords = []
    for e in entries:
        coord = convBoxCoord(e)
        if coord[0] == True: #Only add if box is a person and not a group
            coords.append(coord[1:])
    return coords


def extractPedestrians(imgPath):
    img = cv2.imread(imgPath)
    labels = getLabels(imgPath)
    count = 0
    for l in labels:
        x0, y0, x1, y1 = l
        roi = img[y0:y1, x0:x1]
        fname = os.path.basename(imgPath) + "_" + str(count)
        cv2.imwrite("D:\\Datasets\\ExtractedPed\\train\\" + fname, img)
        count += 1

if __name__ == '__main__':
    imgs = glob.glob("**/*.png")
    p = Pool(4)
    p.map(extractPedestrians, imgs)