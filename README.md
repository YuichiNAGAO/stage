# Implementation for palm tree detection using Yolo-v3 

Reference : https://github.com/eriklindernoren/PyTorch-YOLOv3
 
#### 1. Clone
 ```
 git clone https://github.com/YuichiNAGAO/stage.git
 cd stage
 ```

#### 2. Prepare data

Put all .png and .cnn files in `img_and_cnn` directory  
Write the names of files in `image_list.txt` (Already written)

#### 3. Create table
```
python annotation_process.py
```
You can find tables in `data` directory 

#### 4. Crop images and split data
```
python image_cropping.py
```
 
#### 5. Training and validation
There are towo options to label the bounding box, by adding `--cropping_config [option]` to command line.<br>
In the case `--cropping_config 1` : set the bounding box if all parts of the tree are included in the image. <br>
In the case `--cropping_config 2` : set the bounding box if the center of the tree is in the image.<br><br>
And, there are four options to set classification. By adding `--classes [option]` to command line.<br>
In the case `--classes 0` : Classification problem become Coco+Raphia.  <br>
In the case `--classes 1` : Classification problem become Coco+Raphia+Others.<br>
In the case `--classes 2` : Classification problem become Coco+Others. <br>
In the case `--classes 3` : Classification problem become Raphia+Others.<br>
```
cd weights/
bash download_weights.sh
cd ..
python  train.py --epochs 100 --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74 --cropping_config 2 --classes 1
```
