#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from test_big_image import class2num, get_statistics_big

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import warnings
import json
import cv2
import glob
import numpy as np
import shutil
from tqdm import tqdm
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def arange_dir(path):
    os.makedirs(path, exist_ok=True)
    for file in glob.glob(path+"/*"):
        os.remove(file)

def num2str(num):
    if num==0:
        return "Coco+Raphia"
    elif num==1:
        return "Coco+Raphia+Others"
    elif num==2:
        return "Coco+Others"
    elif num==3:
        return "Raphia+Others"

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="set model name")
    parser.add_argument('--shuffle_train_valid', action='store_true')
    parser.add_argument('--recreate_data', action='store_true')
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--cropping_size", type=int, default=832, help="size of each cropped image dimension")
    parser.add_argument("--cropping_step", type=int, default=208, help="step of cropping process")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--cropping_config",type=int, default=1, help="type of cinfigration of cropping, choose 1 or 2")
    parser.add_argument("--classes",type=int, default=1, help="0:Coco+Raphia, 1:Coco+Raphia+Others, 2:Coco+Others, 3:Raphia+Others")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")

    
    
    opt = parser.parse_args()
    print(opt)
    
    num_class=opt.classes
    str_class=num2str(opt.classes)
    
    
    with open("image_list_test.txt") as f:
        image_test=f.read().strip()
        txt_test=image_test.split(".")[0]

    with open("data/"+txt_test+".txt") as f:
        table = [s.strip().split() for s in f.readlines()[1:]]  


    datasets_big=Image_big('img_and_cnn/'+image_test)

    dataloader_big = DataLoader(
            datasets_big,
            batch_size=1,
            shuffle=False,
            num_workers=opt.n_cpu,
        )
    
    files=sorted(glob.glob('data/custom/images/*'))
    
    with open("image_list_train.txt") as f:
            l_strip_train = [s.strip() for s in f.readlines()]  
    
    with open("image_list_valid.txt") as f:
            l_strip_valid = [s.strip() for s in f.readlines()] 
    
    if opt.recreate_data:
        arange_dir("data/custom/images_with_bb")
        arange_dir("data/custom/images_with_bb_2")
        arange_dir("data/custom/labels_not_yolo")
        arange_dir("data/custom/labels_not_yolo_2")
        arange_dir("data/custom/labels")
        arange_dir("data/custom/images")
        arange_dir("data/custom/labels_2")

        if l_strip_train:
            nb_train=operation(l_strip_train,0,opt.cropping_step,opt.cropping_size)

        if l_strip_valid:
            nb_valid=operation(l_strip_valid,nb_train,opt.cropping_step,opt.cropping_size)
    
    else:
        nb_train=int(len(files)*0.5)

    path_train="data/custom/train.txt"
    path_valid="data/custom/valid.txt"
    if os.path.exists(path_train):
        os.remove(path_train)
    if os.path.exists(path_valid):
        os.remove(path_valid)

    
    if opt.shuffle_train_valid:
        nb_train=int(len(files)*0.75)

        files_shuffled=random.sample(files, len(files))

        for file in files_shuffled[:nb_train]:
            f = open(path_train, 'a') # 書き込みモードで開く
            f.write(file+"\n") # 引数の文字列をファイルに書き込む
            f.close()

        for file in files_shuffled[nb_train:]:
            f = open(path_valid, 'a') # 書き込みモードで開く
            f.write(file+"\n") # 引数の文字列をファイルに書き込む
            f.close()

    if not opt.shuffle_train_valid:

        for file in files[:nb_train]:
            f = open(path_train, 'a') # 書き込みモードで開く
            f.write(file+"\n") # 引数の文字列をファイルに書き込む
            f.close()
        if len(files)>nb_train*1.3:
            for file in files[nb_train:int(nb_train*1.3)]:
                f = open(path_valid, 'a') # 書き込みモードで開く
                f.write(file+"\n") # 引数の文字列をファイルに書き込む
                f.close()
        else:
            for file in files[nb_train:]:
                f = open(path_valid, 'a') # 書き込みモードで開く
                f.write(file+"\n") # 引数の文字列をファイルに書き込む
                f.close()

   
    data_dict=opt.__dict__
    data_dict["training image"]=' '.join(l_strip_train)
    data_dict["validation image"]=' '.join(l_strip_valid)
    data_dict["classes"]=str_class

    with open("./config/models/"+opt.model_name+".json", mode="w") as f:
        json.dump(data_dict, f, indent=4)    
    
    logger = Logger("logs")
    
    log_name=str(opt.cropping_config)+"_"+str(num_class)+"_"+image_test.split('.')[0]+"_"+str(opt.epochs)
    
    log_dir='logs_epoch/'+log_name
    
    os.makedirs(log_dir, exist_ok=True)
    
    logger_2 = Logger(log_dir)
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    arrange_data(num_class,opt.cropping_config)
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    print(class_names)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, opt.cropping_config, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                crop_config=opt.cropping_config,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
        
        if (epoch+1) % (opt.epochs//100) == 0:
            print("\nPerforming object detection in big image:")
            for batch_i, input_imgs in enumerate(tqdm(dataloader_big)):
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
            
            
            if num_class==2:
                table_new=[[0 if line[1]=="Coco" else 1 ,int(int(line[2])/2),int(int(line[4])/2),int(int(line[3])/2),int(int(line[5])/2)] for line in table]
            elif num_class==3:
                table_new=[[1 if line[1]=="Raphia" else 0 ,int(int(line[2])/2),int(int(line[4])/2),int(int(line[3])/2),int(int(line[5])/2)] for line in table]
            elif num_class==1:
                table_new=[[class2num(line[1])  ,int(int(line[2])/2),int(int(line[4])/2),int(int(line[3])/2),int(int(line[5])/2)] for line in table]

            table_new_tensor=torch.Tensor(table_new)            
            
            target_cls =table_new_tensor[:,0].to('cpu').detach().numpy().copy()
                        
            precision_big,recall_big,f1_big,pred_scores,pred_labels,true_positives=get_statistics_big(detections,table_new_tensor,target_cls,opt.iou_thres)                
            
 
            pres, rec, AP_big, f_val, ap_cls=ap_per_class(true_positives,pred_scores,pred_labels,target_cls)
            
            print("precision: ",precision_big)
            print("recall: ",recall_big)
            print("f value: ",f1_big)
            print("mAP: ",AP_big.mean())
        
            evaluation_metrics_big = [
                ("val_precision_big", precision_big),
                ("val_recall_big", recall_big),
                ("val_f1_big", f1_big),
                ("val_mAP_big", AP_big.mean()),
            ]
            
            logger_2.list_of_scalars_summary(evaluation_metrics_big, epoch)
            
        if (epoch+1) % opt.epochs == 0:
            torch.save(model.state_dict(), "checkpoints/{}.pth".format(opt.model_name))
        

