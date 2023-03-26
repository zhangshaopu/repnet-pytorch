import os
import math
import time
import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, ConcatDataset
from IPython.display import clear_output


from trainLoop import running_mean, training_loop, trainTestSplit, plot_grad_flow
from Model_inn import RepNet
from Dataset import getCombinedDataset
from SyntheticDataset import SyntheticDataset
from BlenderDataset import BlenderDataset

from torchinfo import summary

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cpu")

frame_per_vid = 64
multiple = False

testDatasetC = getCombinedDataset('countix/countix_test.csv',
                                   'E:/dataset/testvids',
                                   'test',
                                   frame_per_vid=frame_per_vid,
                                   multiple=multiple
                                )
# testDatasetS = SyntheticDataset('E:/dataset/synthvids', 'train*', 'mp4', 2000,
#                                    frame_per_vid=frame_per_vid
#                                 )

testList = [testDatasetC]
random.shuffle(testList)
testDataset = ConcatDataset(testList)



trainDatasetC = getCombinedDataset('countix/countix_train.csv',
                                   'E:/dataset/trainvids',
                                   'train',
                                   frame_per_vid=frame_per_vid,
                                   multiple=multiple)
#trainDatasetS1 = SyntheticDataset('/home/saurabh/Downloads/HP72','HP72', 'mp4', 500,
#                                   frame_per_vid=frame_per_vid)
#trainDatasetS2 = SyntheticDataset('/home/saurabh/Downloads', '1917', 'mkv', 500,
#                                   frame_per_vid=frame_per_vid)
# trainDatasetS3 = SyntheticDataset('E:/dataset/synthvids', 'train*', 'mp4', 3000,
#                                    frame_per_vid=frame_per_vid)
#trainDatasetS4 = SyntheticDataset('/home/saurabh/Downloads', 'HP6', 'mkv', 500,
#                                   frame_per_vid=frame_per_vid)
# #trainDatasetB = BlenderDataset('E:/dataset/blendervids', 'videos', 'annotations', frame_per_vid)
#test
trainList = [trainDatasetC] #, trainDatasetB]
random.shuffle(trainList)
trainDataset = ConcatDataset(trainList)

model =  RepNet(frame_per_vid)
# summary(model,input_size = (1, 64, 3, 112, 112)) # batchsize = 1 , frame = 64 , 112 * 112 * 3 
model = model.to(device)

print("done")

"""Testing the training loop with sample datasets"""
 
sampleDatasetA = torch.utils.data.Subset(trainDataset, range(0, len(trainDataset)))
sampleDatasetB = torch.utils.data.Subset(testDataset, range(0,  len(testDataset)))

print("len(sampleDatasetA):" ,len(sampleDatasetA)) #4332
print("len(sampleDatasetB):" ,len(sampleDatasetB)) #

if __name__ == '__main__':
    trLoss, valLoss = training_loop(  10,
                                    model,
                                    sampleDatasetA,
                                    sampleDatasetB,
                                    1,
                                    6e-5,
                                    'x3dbb',
                                    use_count_error=False,
                                    saveCkpt = 1,
                                    train = 1,
                                    validate = 1,
                                    lastCkptPath = None #'checkpoint/blender_no_mha_yes5.pt'
                                    )