import glob
import random
import cv2
import os
import torchvision.transforms as transforms
import torch
import random
import pickle
from torch.utils.data import Dataset
from PIL import Image
import numpy as np 

desired_im_sz = (128, 128)

def to_rgb(image):
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)
        return rgb_image

def random_crop_np(hrand, wrand, imag) :
        h = int(np.round(len(imag)*hrand))
        w = int(np.round(len(imag[0])*wrand))
        imag = imag[h:h+desired_im_sz[0],w:w+desired_im_sz[1],:]
        return imag

class ImageTestDataset(Dataset):

    def __init__(self, video_pkl_file, transforms_=None,nt=20):
        self.video_pkl_file = video_pkl_file
        self.transform = transforms.Compose(transforms_)
        self.nt = nt
        test_pkl_file = open(video_pkl_file, 'rb')
        self.data = pickle.load(test_pkl_file)
     
    def __getitem__(self, index):
        frame_seq = []
        h_rand = random.uniform(0,0.91)
        w_rand = random.uniform(0,0.73)
        for img_name in self.data[index]:
            f = open(img_name, 'rb')
            img = pickle.load(f)
            f.close()
            img = random_crop_np(h_rand, w_rand, img)
            img = cv2.resize(img, (64, 64))
            img[:,:,0] = img[:,:,0] + img[:,:,2]
            img[:,:,1] = img[:,:,1] + img[:,:,2]
            img = (img/np.ndarray.max(img))
            img = img.astype(np.float32)
            img = np.moveaxis(img, 2,0)
            img = torch.from_numpy(img)
            img = self.transform(img)
            frame_seq.append(img)
        frame_seq = torch.stack(frame_seq, 0).permute(0,3,2,1)
        return frame_seq
  
    def __len__(self):
        return len(self.data)