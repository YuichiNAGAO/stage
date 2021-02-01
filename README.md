# Implementation for palm tree detection using Yolo-v3 

Reference : https://github.com/eriklindernoren/PyTorch-YOLOv3
 
#### 1. Clone
 ```
 git clone https://github.com/YuichiNAGAO/stage.git
 cd stage
 ```

#### 2. Prepare data

Put all .png and .cnn files in `img_and_cnn` directory.  <br>
Write the names of files in `image_list_table.txt`. <br>
The contents of `image_list.table.txt` would be like
 ```
 Sce_CocoRaph500_All_SIS_H.png Sce_CocoRaph500_Cl_SI_H.png Sce_CocoRaph500_Cl_Segv1.cnn
Sce_CocoRaph900H_All_SIS_H.png Sce_CocoRaph900H_All_Cl_H.png Sce_CocoRaph900H_All_Cl_H_Segv1.cnn
Sce_CocoRaphBig630_All_WS_SI_H.png Sce_CocoRaphBig630_Cl_SI_H.png Sce_CocoRaphBig630_Cl_SI_H_Segv1.cnn
Sce_CocoRaphBig400_HRA_All_WS_SI_H.png Sce_CocoRaphBig400_HRA_Cl_SI_H.png Sce_CocoRaphBig400_HRA_Cl_SI_H_Segv1.cnn
 ```

#### 3. Create table
```
python annotation_process.py
```
You can find tables in `data` directory.
It takes 5 minutes (small image) to 20 minutes (big image) to complete the operation.<br>
Once you run this code, you don't have to run it again when you run other model. The table is permanently stored in `data` directory.

#### 4. Training and validation
- Write the names of files which you want to use as training data and validation data in `image_list_train.txt` and `image_list_valid.txt`.<br>
The contents of `image_list_train.txt` would be like
```
Sce_CocoRaph900H_All_SIS_H.png 
```
The contents of `image_list_valid.txt` would be like
```
 Sce_CocoRaph500_All_SIS_H.png 
```
- If you want to shuffle training data and testing data, put `--shuffle` at the end of the code. <br><br>
- There are two options to label the bounding box, by adding `--cropping_config [option]` to command line.<br>
In the case `--cropping_config 1` : set the bounding box if all parts of the tree are included in the image. <br>
In the case `--cropping_config 2` : set the bounding box if the center of the tree is in the image.<br><br>
- There are four options to set classification. By adding `--classes [option]` to command line.<br>
In the case `--classes 0` : Classification problem become Coco+Raphia.  <br>
In the case `--classes 1` : Classification problem become Coco+Raphia+Others.<br>
In the case `--classes 2` : Classification problem become Coco+Others. <br>
In the case `--classes 3` : Classification problem become Raphia+Others.<br>
- You have to write model name in `--model_name [model name]`

```
cd weights/
bash download_weights.sh
cd ..
python  train.py --model_name [model name] --epochs 100 --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74 --cropping_config 2 --classes 1
```

#### 5. Testing
Write the names of files which you want to use as testing data in `image_list_test.txt`.<br>
The contents of `image_list_test.txt` would be like
```
Sce_CocoRaph900H_All_SIS_H.png 
```
Run the code with model name that you have created in the last step.
```
python test_big_image.py --model_name [model name]
```
#### 6. Result
You can check the trace of training and validation by Tensorboard.
```
tensorboard --logdir='logs' --port=6006
```
Go to http://localhost:6006/
