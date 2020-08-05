from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import cv2
import os
import sys
import time
import datetime
import argparse
from tqdm import tqdm
import json
import numpy as np

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def class2num(class_name):
    if class_name=="Coco":
        return 0
    elif class_name=="Raphia":
        return 1
    else:
        return 2

def str2num(classes):
    if classes== "Coco+Raphia":
        return 0
    elif classes== "Coco+Raphia+Others":
        return 1
    elif classes== "Coco+Others":
        return 2
    elif classes=="Raphia+Others" :
        return 3
    
def get_statistics_big(outputs, targets,target_cls, iou_threshold):
    output = outputs[0]
    pred_boxes = output[:, :4]
    pred_scores = output[:, 4]
    pred_labels = output[:, -1]
    target_boxes = targets[:, 1:]
    target_labels = targets[:, 0]
    num_pos=pred_boxes.shape[0]
    num_true=target_boxes.shape[0]
    detected_boxes = []
    true_positives = np.zeros(pred_boxes.shape[0])
    for pred_i,(pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        # If targets are found break
        if len(detected_boxes) == len(targets):
            break

        # Ignore if label is not one of the target labels
        if pred_label not in target_labels:
            continue

        iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
        if iou >= iou_threshold and box_index not in detected_boxes:
            true_positives[pred_i] = 1
            detected_boxes += [box_index]
    sum_tp=int(np.sum(true_positives))
    precision= sum_tp/num_pos
    recall=sum_tp/num_true
    f=2*precision*recall/(precision+recall)
    return precision,recall,f,pred_scores,pred_labels,true_positives

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="which model you use")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--cropping_config",type=int, default=1, help="type of cinfigration of cropping, choose 1 or 2")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    with open("./config/models/"+opt.model_name+".json", mode="r") as f:
            df = json.load(f)    
            classes_type=str2num(df['classes'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)
    # Set up model
    model = Darknet(df['model_def'], img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load("checkpoints/{}.pth".format(opt.model_name)))
    model.eval()

    step=208*3
    crop_size=832 

    with open("image_list_test.txt") as f:
        image=f.read().strip()
        txt=image.split(".")[0]

    with open("data/"+txt+".txt") as f:
        table = [s.strip().split() for s in f.readlines()[1:]]  


    datasets_big=Image_big('img_and_cnn/'+image)

    dataloader = DataLoader(
            datasets_big,
            batch_size=1,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

    classes = load_classes("data/custom/classes.names")  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    for batch_i, input_imgs in enumerate(tqdm(dataloader)):
        input_imgs = Variable(input_imgs.type(Tensor))
        y=batch_i//datasets_big.nb_image_w
        x=batch_i%datasets_big.nb_image_w
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections[...,0]+=312* x/datasets_big.ratio_w
            detections[...,1]+=312* y/datasets_big.ratio_h
            detections[...,2]/= datasets_big.ratio_w
            detections[...,3]/= datasets_big.ratio_h
        if not batch_i:
            total_detections=detections
        else:
            total_detections=torch.cat((total_detections,detections),1)


    detections = non_max_suppression(total_detections, opt.conf_thres, opt.nms_thres)


    if classes_type==2:
        table_new=[[0 if line[1]=="Coco" else 1 ,int(int(line[2])/2),int(int(line[4])/2),int(int(line[3])/2),int(int(line[5])/2)] for line in table]
    elif classes_type==3:
        table_new=[[1 if line[1]=="Raphia" else 0 ,int(int(line[2])/2),int(int(line[4])/2),int(int(line[3])/2),int(int(line[5])/2)] for line in table]
    elif  classes_type==1:
        table_new=[[class2num(line[1])  ,int(int(line[2])/2),int(int(line[4])/2),int(int(line[3])/2),int(int(line[5])/2)] for line in table]

    table_new_tensor=torch.Tensor(table_new)            

    precision,recall,f=get_statistics_big(detections,table_new_tensor,opt.iou_thres)    


    dict_result={"model":df, "test image":image ,"precision":precision,"recall":recall,"f value":f}

    print("precision: ",precision)
    print("recall: ",recall)
    print("f value: ",f)

    with open("./output/"+opt.model_name+"_"+image+".json", mode="w") as f:
            json.dump(dict_result, f, indent=4)    

    print("Result is saved in {}".format("output/"+opt.model_name+"_"+image+".json"))            








