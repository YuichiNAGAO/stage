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
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import csv

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def col_row_name(classes_type):
    if classes_type==2:
        return ["Coco","Others"]
    
    elif classes_type==3:
        return ["Others","Raphia"]
        



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
    output = outputs[0]#torch.Size([864, 7])
    pred_boxes = output[:, :4]#torch.Size([864, 4])
    pred_scores = output[:, 4]#torch.Size([864])
    pred_labels = output[:, -1]#torch.Size([864])
    target_boxes = targets[:, 1:]#torch.Size([986, 4])
    target_labels = targets[:, 0]#torch.Size([986])
    num_pos=pred_boxes.shape[0]#864
    num_true=target_boxes.shape[0]#986
    detected_boxes = []
    true_positives = np.zeros(pred_boxes.shape[0])#(864,)
    i = np.argsort(-pred_scores)
    pred_boxes, pred_labels,pred_scores=pred_boxes[i],pred_labels[i],pred_scores[i]
    
    
    
    for pred_i,(pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        # If targets are found break
        if len(detected_boxes) == len(targets):
            break

        # Ignore if label is not one of the target labels
        if pred_label not in target_labels:
            continue

        iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)#torch.Size([1, 4]),torch.Size([986, 4]),(xmin,ymin,xmax.ymax)
        if iou >= iou_threshold and box_index not in detected_boxes:
            if target_labels[box_index]==pred_label:
                true_positives[pred_i] = 1
                detected_boxes += [box_index]
                #target_boxes=torch.cat([target_boxes[0:box_index], target_boxes[box_index+1:]])
            else:
                detected_boxes += [box_index]
                if true_positives[pred_i] != 1:
                    true_positives[pred_i] = -1
            
    return true_positives, pred_scores, pred_labels,detected_boxes

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

   
    try:
        crop_size=df["cropping_size"]
    except KeyError:
        crop_size=832 
       
    red_ratio=float(crop_size/opt.img_size)

    with open("image_list_test.txt") as f:
        image=f.read().strip()
        txt=image.split(".")[0]
    
    with open("data/"+txt+".txt") as f:
        table = [s.strip().split() for s in f.readlines()[1:]]  


    datasets_big=Image_big('img_and_cnn/'+image,crop_size=crop_size)

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
    
    step_size=int(opt.img_size*3/4)

    print("\nPerforming object detection:")
    for batch_i, input_imgs in enumerate(tqdm(dataloader)):
        input_imgs = Variable(input_imgs.type(Tensor))
        y=batch_i//datasets_big.nb_image_w
        x=batch_i%datasets_big.nb_image_w
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections[...,0]+=step_size* x/datasets_big.ratio_w
            detections[...,1]+=step_size* y/datasets_big.ratio_h
            detections[...,2]/= datasets_big.ratio_w
            detections[...,3]/= datasets_big.ratio_h
        if not batch_i:
            total_detections=detections
        else:
            total_detections=torch.cat((total_detections,detections),1)


    detections = non_max_suppression(total_detections, opt.conf_thres, opt.nms_thres)


    if classes_type==2:
        table_new=[[0 if line[1]=="Coco" else 1 ,int(int(line[2])/red_ratio),int(int(line[4])/red_ratio),int(int(line[3])/red_ratio),int(int(line[5])/red_ratio)] for line in table]
    elif classes_type==3:
        table_new=[[1 if line[1]=="Raphia" else 0 ,int(int(line[2])/red_ratio),int(int(line[4])/red_ratio),int(int(line[3])/red_ratio),int(int(line[5])/red_ratio)] for line in table]
    elif  classes_type==1:
        table_new=[[class2num(line[1])  ,int(int(line[2])/red_ratio),int(int(line[4])/red_ratio),int(int(line[3])/red_ratio),int(int(line[5])/red_ratio)] for line in table]

    table_new_tensor=torch.Tensor(table_new)            
    
    target_cls =table_new_tensor[:,0].to('cpu').detach().numpy().copy()
    
    
    true_positives, pred_scores, pred_labels,detected_boxes=get_statistics_big(detections,table_new_tensor,target_cls,opt.iou_thres)                

    
    #true_positives (864,) array
    #pred_scores torch.Size([864])
    #pred_labels torch.Size([864])
    #target_cls (986,) array
    
    pres, rec, AP_big, f_val, ap_cls,false_pos_vacant,true_pos,false_pos,ground_truth,num_predict=ap_per_class(true_positives,pred_scores,pred_labels,target_cls,detected_boxes)    
    false_neg=[ground_truth[0]-true_pos[0]-false_pos[1],ground_truth[1]-true_pos[1]-false_pos[0]]
    
    
    matrix_confusion=[[col_row_name(classes_type)[0],true_pos[0],false_pos[1],false_neg[0],ground_truth[0]], [col_row_name(classes_type)[1],false_pos[0],true_pos[1],false_neg[1],ground_truth[1]],["None",false_pos_vacant[0],false_pos_vacant[1],"",""],["Total",num_predict[0],num_predict[1],"",""]]
    table = plt.table(cellText=matrix_confusion,
                      colLabels=['Class',col_row_name(classes_type)[0], col_row_name(classes_type)[1],"None","Total"],
                      cellLoc='center',
                      loc='center')
    table.set_fontsize(25)
    table.scale(1.5, 3)


    # グラフとセットで表示されるので、表は削除
    plt.axis('off')

    os.makedirs("tables", exist_ok=True)
    #保存
    plt.savefig("tables/{}.png".format(opt.model_name+"_"+txt+"_"+"nms"+str(opt.nms_thres).split(".")[1]+"_iou_"+str(opt.iou_thres).split(".")[1]), bbox_inches="tight")
               
    dict_result={"model":df, "test image":image ,"precision":pres.mean(),"recall":rec.mean(),"f value":f_val.mean(),"mAP":AP_big.mean()}

    print("precision: ",pres.mean())
    print("recall: ",rec.mean())
    print("f value: ",f_val.mean())
    print("mAP: ",AP_big.mean())
    
    list_res=["nms_"+str(opt.nms_thres).split(".")[1],"iou_"+str(opt.iou_thres).split(".")[1],pres.mean(),rec.mean(),f_val.mean(),AP_big.mean()]
    
    os.makedirs("results/{}".format(txt), exist_ok=True)
    
    with open("results/{}/{}.csv".format(txt,opt.model_name), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(list_res)
    
    

    with open("./output/"+opt.model_name+"_"+txt+"_"+"nms"+str(opt.nms_thres).split(".")[1]+"_iou_"+str(opt.iou_thres).split(".")[1]+".json", mode="w") as f:
            json.dump(dict_result, f, indent=4)    

    print("Result is saved in {}".format("output/"+opt.model_name+"_"+txt+".json"))            








