import cv2
import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images

def test_image_resize(img,nb_h,nb_w,step=624, crop_size=832 ,img_size=416):
    
    img=cv2.resize(img , (nb_h*step+108, nb_w*step+108))
  
    


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

    
class Image_big(Dataset):
    def __init__(self, folder_path, crop_size=832 ,img_size=416):
        self.step=int(crop_size*3/4)
        self.img_size_3_4=int(img_size*3/4)
        self.residual=int(crop_size*1/4)
        self.crop_size=crop_size
        self.img=cv2.imread(folder_path)
        self.nb_image_h=self.img.shape[0]//self.step
        self.nb_image_w=self.img.shape[1]//self.step
        self.img_new=cv2.resize(self.img , (self.nb_image_w*self.step+self.residual,self.nb_image_h*self.step+self.residual))
        self.ratio_h=self.img_new.shape[0]/self.img.shape[0]
        self.ratio_w=self.img_new.shape[1]/self.img.shape[1]
        self.nb_images=self.nb_image_h*self.nb_image_w
        self.img_size = img_size

    def __getitem__(self, index):
        num=index % self.nb_images
        y=num//self.nb_image_w
        x=num%self.nb_image_w
        input_imgs=np.copy(self.img_new[y*self.step:y*self.step+self.crop_size,x*self.step:x*self.step+self.crop_size])
        input_imgs = cv2.resize(input_imgs , (self.img_size, self.img_size))
        input_imgs = cv2.cvtColor(input_imgs, cv2.COLOR_BGR2RGB)
        input_imgs = transforms.ToTensor()(input_imgs )
        # Extract image as PyTorch tensor
        
        return input_imgs

    def __len__(self):
        return self.nb_images

class ListDataset(Dataset):
    def __init__(self, list_path, crop_config, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        if crop_config==1:
            self.label_files = [
                path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in self.img_files
            ]
        elif crop_config==2:
            self.label_files = [
                path.replace("images", "labels_2").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in self.img_files
            ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):#リスト化されたバッチデータを引数にとる
        paths, imgs, targets = list(zip(*batch))#このアスタリスクはリストの外側の括弧を外すためのもの,zipの引数はタプル型である必要がある
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

