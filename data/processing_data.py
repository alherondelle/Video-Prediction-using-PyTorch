import glob
import random
import os
import torchvision.transforms as transforms
import torch
import random
import pickle
from torch.utils.data import Dataset
from PIL import Image

def split_dataset4KITTI(dataset_path):
    
    #create the folder to save the pickle file..
    if not os.path.exists("data/METEOSAT/pkl/"):
        os.mkdir("data/METEOSAT/pkl/")

    train_video_path = os.path.join(dataset_path,"train")
    train_data = video_load(train_video_path)
    train_output = open('data/METEOSAT/pkl/train_data.pkl', 'wb')
    pickle.dump(train_data, train_output)

    val_video_path = os.path.join(dataset_path,"val")
    val_data = video_load(val_video_path)
    val_output = open('data/METEOSAT/pkl/val_data.pkl', 'wb')
    pickle.dump(val_data, val_output)


    test_video_path = os.path.join(dataset_path,"test")
    test_data = video_load(test_video_path)
    test_output = open('data/METEOSAT/pkl/test_data.pkl', 'wb')
    pickle.dump(test_data, test_output)

def process_data():
    c_dir = 'data/METEOSAT/'
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = {v :[] for v in os.listdir(os.path.join(c_dir, 'val'))}
    splits['test'] = {t :[] for t in os.listdir(os.path.join(c_dir, 'test'))}
    splits['train'] = {r :[] for r in os.listdir(os.path.join(c_dir, 'train'))}

    for split in splits:
        print("@@@@",split)
        im_list = []
        source_list = []  # corresponds to recording that image came from
        #print(len(splits[split]))
        for folder in splits[split]:
            im_dir = os.path.join(c_dir, split, folder)
            _, _, files = os.walk(im_dir).__next__()
            im_list += [im_dir + '/'+f for f in sorted(files)]
            source_list += [folder] * len(files)
            
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    img = Image.fromarray(im)
    prop = int(np.round(target_ds * im.shape[1]))
    img = img.resize((prop,desired_sz[0]))    
    img = np.array(img)
    d = (img.shape[1] - desired_sz[1]) // 2
    img = img[:, d:d+desired_sz[1]]
    return img

def video_load(video_path,nt=8):
    video_info = []
    video_folder = os.listdir(video_path)
    cnt=0
    for video_name in video_folder:
        cnt += 1
        img_list = os.listdir(os.path.join(video_path,video_name))
        img_list.sort()
        num_img = len(img_list)
        for j in range(num_img-nt+1):
            index_set = []
            for k in range(j, j + nt):
                index_set.append(os.path.join(os.path.join(video_path,video_name),img_list[k]))
            video_info.append(index_set)
    return video_info
