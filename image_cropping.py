
import cv2
import os
import glob
import numpy as np
import shutil
from tqdm import tqdm
import random



def save_dict_to_file(dic,filename):
    f = open(filename,'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(path):
    f = open(path,'r')
    data=f.read()
    f.close()
    return eval(data)

def arange_dir(path):
    os.makedirs(path, exist_ok=True)
    for file in glob.glob(path+"/*"):
        os.remove(file)



def save_list(obj,filename):
    f = open(filename,'w')
    for i in obj:
        for j in i:
            f.write(str(j)+" ")
        f.write("\n")
    f.close()
    
with open("image_list.txt") as f:
    l_strip = [s.strip() for s in f.readlines()]    

arange_dir("data/custom/images_with_bb")
arange_dir("data/custom/labels_not_yolo")
arange_dir("data/custom/labels")
arange_dir("data/custom/images")

step=208
size=832    
    
for image_cnn in l_strip:
    print("==========Processing {} now...===========".format(image_cnn.split()[0]) )
    file_png="img_and_cnn/"+image_cnn.split()[0]
    file_cnn="img_and_cnn/"+image_cnn.split()[2]
    img = cv2.imread(file_png)
    img_raw = cv2.imread(file_png)
    name_file="data/"+image_cnn.split()[0].split('_')[0]+'_'+image_cnn.split()[0].split('_')[1]+"_table.txt"
    with open(name_file) as f:
        table = [s.strip().split() for s in f.readlines()[1:]]  
    height=img.shape[0]
    width=img.shape[1] 
    nb_image_h=height//step
    nb_image_w=width//step
    number_im=(nb_image_h-3)*(nb_image_w-3)
    position_cropped=[[[] for j in range(nb_image_w-3)] for i in range(nb_image_h-3)]
    position_cropped_yolo=[[[] for j in range(nb_image_w-3)] for i in range(nb_image_h-3)]
    for tree in table:
        c1, c2 = (int(tree[2]), int(tree[4])), (int(tree[3]), int(tree[5]))
        if tree[1]=="Raphia":
            color=(255, 0, 0)
        else:
            color=(0, 0, 255)
        cv2.rectangle(img, c1, c2, color)
        t_size = cv2.getTextSize(tree[1], 0, fontScale=1 / 3, thickness=1)[0]
        c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, tree[1], (c1[0], c1[1] + t_size[1]), 0, 1 / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)        
    cv2.imwrite(file_png.replace('.png','_remade.png'),img)
    
    print("Creating cropped table...")
    for tree in tqdm(table):
        x_center=(int(tree[2])+int(tree[3]))//2
        y_center=(int(tree[4])+int(tree[5]))//2
        if x_center>(nb_image_w*step) or y_center>(nb_image_h*step):
            continue
        x_cen=x_center//step
        y_cen=y_center//step
    
        for j_x in range(4):
            for j_y in range(4):
                y_pos=y_cen-j_y
                x_pos=x_cen-j_x
                if y_pos<0 or x_pos<0 or y_pos>=(nb_image_h-3) or x_pos>=(nb_image_w-3):
                    continue 
                x_min=int(tree[2])-step*x_pos
                x_max=int(tree[3])-step*x_pos
                y_min=int(tree[4])-step*y_pos
                y_max=int(tree[5])-step*y_pos
                if x_min<0:
                    x_min=0
                if y_min<0:
                    y_min=0
                if x_max>832:
                    x_max=size
                if y_max>832:
                    y_max=size
                position_cropped[y_pos][x_pos].append([tree[1],x_min,x_max,y_min,y_max])  
          
    print("Creating Yolo annotation...")  
    for i,row in tqdm(enumerate(position_cropped)):
        for j, col in enumerate(row):
            number=str(i*(nb_image_w-3)+j).zfill(4)
            filename="data/custom/labels_not_yolo/image_"+number+".txt"
            save_list(col,filename)
            for obj in col:
                if obj[0]=="Coco":
                    n_class=0
                else:
                    n_class=1
                position_cropped_yolo[i][j].append([n_class,float((obj[1]+obj[2])/2/size),float((obj[3]+obj[4])/2/size),float((obj[2]-obj[1])/size),float((obj[4]-obj[3])/size)])
#labels        
    print("Saving Yolo annotation...")  
    for i,row in tqdm(enumerate(position_cropped_yolo)):
        for j,img_pos in enumerate(row):
            number=str(i*(nb_image_w-3)+j).zfill(4)
            filename="data/custom/labels/image_"+number+".txt"

            save_list(img_pos,filename)
#images   
    print("Saving images...")  
    for y in tqdm(range(nb_image_h-3)):
        for x in range(nb_image_w-3):
            number=str(y*(nb_image_w-3)+x).zfill(4)
            im1=np.copy(img_raw[y*step:y*step+size,x*step:x*step+size])
            im2=np.copy(img_raw[y*step:y*step+size,x*step:x*step+size])
            im1 = cv2.resize(im1 , (416, 416))
            cv2.imwrite("data/custom/images/image_"+number+".png",im1)
            for i in position_cropped[y][x]:
                c1, c2 = (i[1], i[3]), (i[2], i[4])
                if i[0]=="Raphia":
                    color=(255, 0, 0)
                else:
                    color=(0, 0, 255)

                cv2.rectangle(im2, c1, c2, color)
                t_size = cv2.getTextSize(i[0], 0, fontScale=1 / 3, thickness=1)[0]
                c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
                cv2.rectangle(im2, c1, c2, color, -1, cv2.LINE_AA)
                cv2.putText(im2, i[0], (c1[0], c1[1] + t_size[1]), 0, 1 / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.imwrite("data/custom/images_with_bb/image_bb_"+number+".png",im2) 
            

path_train="data/custom/train.txt"
path_valid="data/custom/valid.txt"
if os.path.exists(path_train):
    os.remove(path_train)
if os.path.exists(path_valid):
    os.remove(path_valid)

files=glob.glob('data/custom/images/*')

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
    
    