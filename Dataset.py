import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import cv2
import glob
from tqdm import tqdm
import random
from random import randrange, randint
import math, base64, io, os, time
from torch.utils.data import Dataset, DataLoader, ConcatDataset


"""Creates one sequence from each video"""
class miniDataset(Dataset):
    
    def __init__(self, df, path_to_video):
        
        self.path = path_to_video
        self.df = df.reset_index()
        self.count = self.df.loc[0, 'count']
        self.fps = ""
        self.numFrames = ""
        

    def getFrames(self, path = None):
        """returns frames"""
    
        frames = []
        if path is None:
            path = self.path
        
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(" file: " , path)
        print(" fps: ", self.fps)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            
            img = Image.fromarray(frame)
            frames.append(img)
        
        cap.release()
        
        return frames

    def __getitem__(self, index):
        
        curFrames = self.getFrames()
        self.numFrames = len(curFrames)
        Xlist = []
        for img in curFrames:
            preprocess = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            frameTensor = preprocess(img).unsqueeze(0)
            Xlist.append(frameTensor)
        

        stride = max(self.numFrames // 64 , 1)
        X = Xlist[::stride][:64]
        if len(X) < 64:
            last_frame = Xlist[-1]
            num_pad_frames = 64 - len(X)
            pad_frames = torch.repeat_interleave(last_frame, num_pad_frames, dim=0)
            X = np.concatenate((torch.cat(X), pad_frames), axis=0)
        X = torch.cat(X)
        # X = torch.cat(Xlist) # (numFrames , 3, 112, 112)

        y1 = torch.zeros(64)   
        y1[:] = self.count 

        y2 = torch.zeros((64 , 1))
        y2[:, ] = 1 # (64 , 1)

        # y.extend([output_len/self.count if 1<output_len/self.count<32 else 0 for i in range(0, output_len)])
        
        # y.extend( [ 0 for i in range(0, b)] )
        # y = torch.FloatTensor(y).unsqueeze(-1)
        
        return X, y1, y2
        
    def __len__(self):
        return 1
    
class dataset_with_indices(Dataset):

    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        X, y = self.ds[index]
        return X, y, index
    
    def getPeriodDist(self):
        arr = np.zeros(32,)
        
        for i in tqdm(range(self.__len__())):
            _, p,_ = self.__getitem__(i)
            per = max(p)
            arr[per] += 1
        return arr
    
    def __len__(self):
        return len(self.ds)


def getCombinedDataset(dfPath, videoDir, videoPrefix,frame_per_vid,multiple):
    df = pd.read_csv(dfPath)
    path_prefix = videoDir + '/' + videoPrefix
    
    files_present = []
    for i in range(0, len(df)):
        path_to_video = path_prefix + str(i) + '.mp4'
        if os.path.exists(path_to_video):
            files_present.append(i)

    df = df.iloc[files_present]
    
    miniDatasetList = []
    for i in range(0, len(df)):
        dfi = df.iloc[[i]]
        path_to_video = path_prefix + str(dfi.index.item()) +'.mp4'
        miniDatasetList.append(miniDataset(dfi, path_to_video))
        
    megaDataset = ConcatDataset(miniDatasetList)
    return megaDataset
