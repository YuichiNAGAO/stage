
# -*- coding: utf-8 -*-


import cv2
from tqdm import tqdm
import os
import math

def make_empty_dic():
    dic={}
    dic["type"]=None
    dic["x_min"]=None
    dic["x_max"]=None
    dic["y_min"]=None
    dic["y_max"]=None
    dic["nbpx"]=0
    dic["sum_x"]=0
    dic["sum_y"]=0
    dic["sum_x_x"]=0
    dic["sum_y_y"]=0
    return dic

def save_list(obj,filename):
    if os.path.exists(filename):
        os.remove(filename)
    f = open(filename,'w')
    f.write("No -class -x_min -x_max -y_min -y_max -x_moy -y_moy -x_ecart -y_ecart -red -blue -green\n")
    for i in obj:
        for j in i:
            f.write(str(j)+" ")
        f.write("\n")
    f.close()
    
with open("image_list_table.txt") as f:
    l_strip = [s.strip() for s in f.readlines()]    
    
Nmax = 256 * 256 * 256

    
for image_cnn in l_strip:
    print("==========Processing {} now...===========".format(image_cnn.split()[1]) )
    file_png="img_and_cnn/"+image_cnn.split()[1]
    file_cnn="img_and_cnn/"+image_cnn.split()[2]
    img = cv2.imread(file_png)
    name_file=image_cnn.split()[0].split('_')[0]+'_'+image_cnn.split()[0].split('_')[1]+"_table.txt"
    table_data=[]
    util=[0 for _ in range(Nmax+1)]
    for _ in range(Nmax+1):
        dic=make_empty_dic()
        table_data.append(dic)
    with open(file_cnn) as f:
        list_cnn = [s.strip() for s in f.readlines()]
    list_cnn=list_cnn[2:]
    for i in list_cnn:        
        if "Coco" in i.split()[2] or "coco" in i.split()[2] or "COCO" in i.split()[2] or "cOCo" in i.split()[2] :
            nb=int(i.split()[3])+int(i.split()[4])*256+int(i.split()[5])*256*256
            table_data[nb]["type"]="Coco"
            table_data[nb]["red"]=i.split()[3]
            table_data[nb]["green"]=i.split()[4]
            table_data[nb]["blue"]=i.split()[5]            
            util[nb]=1
        elif "Raphia" in i.split()[2] or "raphia" in i.split()[2] or "RAPHIA" in i.split()[2]:
            nb=int(i.split()[3])+int(i.split()[4])*256+int(i.split()[5])*256*256
            table_data[nb]["type"]="Raphia"
            table_data[nb]["red"]=i.split()[3]
            table_data[nb]["green"]=i.split()[4]
            table_data[nb]["blue"]=i.split()[5]  
            util[nb]=1
        else:
            nb=int(i.split()[3])+int(i.split()[4])*256+int(i.split()[5])*256*256
            table_data[nb]["type"]="Others"
            table_data[nb]["red"]=i.split()[3]
            table_data[nb]["green"]=i.split()[4]
            table_data[nb]["blue"]=i.split()[5]  
            util[nb]=1
            
    height=img.shape[0]
    width=img.shape[1]   
    for i in tqdm(range(height)):
        for j in range(width): 
            nb=img[i][j][2]+img[i][j][1]*256+img[i][j][0]*256*256
            if  util[nb]:
                table_data[nb]["nbpx"]+=1
                table_data[nb]["sum_x"]+=j
                table_data[nb]["sum_y"]+=i
                table_data[nb]["sum_x_x"]+=j*j
                table_data[nb]["sum_y_y"]+=i*i
                if table_data[nb]["x_min"]==None:
                    table_data[nb]["x_min"]=j
                    table_data[nb]["x_max"]=j
                    table_data[nb]["y_min"]=i
                    table_data[nb]["y_max"]=i
                else:
                    if j<table_data[nb]["x_min"]:
                        table_data[nb]["x_min"]=j
                    if j>table_data[nb]["x_max"]:
                        table_data[nb]["x_max"]=j
                    if i<table_data[nb]["y_min"]:
                        table_data[nb]["y_min"]=i
                    if i>table_data[nb]["y_max"]:
                        table_data[nb]["y_max"]=i
    poped_list=[ dic for dic in table_data if not dic["x_min"]==None ]
    
    big_table=[]
    for i,dic in enumerate(poped_list):
        mean_x=int(dic["sum_x"]/dic["nbpx"])
        mean_y=int(dic["sum_y"]/dic["nbpx"])
        std_x=math.sqrt(dic["sum_x_x"]/dic["nbpx"]-mean_x**2)
        std_y=math.sqrt(dic["sum_y_y"]/dic["nbpx"]-mean_y**2)
        small_table=[i,dic["type"],dic["x_min"],dic["x_max"],dic["y_min"],dic["y_max"],mean_x,mean_y,round(std_x, 1),round(std_y,1),dic["red"],dic["green"],dic["blue"]]
        big_table.append(small_table)
    
    
            
                  
    save_list(big_table,"./data/"+name_file)
                    
                    
            
