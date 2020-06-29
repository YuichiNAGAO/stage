# Implementation　for palm tree detectionN　using Yolo-v3 （リポジトリ/プロジェクト/OSSなどの名前）

Reference : https://github.com/eriklindernoren/PyTorch-YOLOv3
 
#### 1.Clone
 ```
 git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
 cd stage
 ```

#### 2.Prepare data

Put all .png and .cnn files in `img_and_cnn` directory
Write the names of files in `image_list.txt`

#### 3.Create table
```
python annotation_process.py
```

#### 4.Crop images and split data
```
python image_cropping.py
```
 
#### 5.Training
```
cd weights/
bash download_weights.sh
cd ..
python  train.py --epochs 1000 --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74
```
