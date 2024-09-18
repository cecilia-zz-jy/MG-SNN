# Imports
from __future__ import division

import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from dataloader import UvaDataset
from utils import save_gray_image
import torch.nn.functional as F
# if you have a CUDA-enabled GPU the set the GPU flag as True
GPU=False

import time
import numpy as np
if GPU:
    import cupy as cp  # You need to have a CUDA-enabled GPU to use this package!
else:
    cp=np

# Parameter setting
# thr = [20] 
thr = [30]  # The threshold of hidden and output neurons 
lr = [0.02]  # The learning rate of hidden and ouput neurons
lamda = [0.000001]  # The regularization penalty for hidden and ouput neurons
b = [1]  # The upper bound of wight initializations for hidden and ouput neurons
a = [-1]  # The lower bound of wight initializations for hidden and ouput neurons
Nepoch = 20  # The maximum number of training epochs
NumOfClasses = 120*100 # Number of classes
Nlayers = 1  # Number of layers
Dropout = [1]
tmax = 256  # Simulation time
GrayLevels = 255  # Image GrayLevels

# General settings
loading = True  # Set it as True if you want to load a pretrained model
LoadFrom = "/data/Weight/weights_mae_best.npy"  # The pretrained model
saving = False
mae_best = 0
Nnrn = [NumOfClasses]  # Number of neurons at hidden and output layers


# MAE 
trainMaeLoss = []
testMaeLoss = []

W = []  # To hold the weights of hidden and output layers
firingTime = []  # To hold the firing times of hidden and output layers
Spikes = []  # To hold the spike trains of hidden and output layers
X = []  # To be used in converting firing times to spike trains
firingTarget = cp.zeros([NumOfClasses])  # To keep the firingTarget firing times of current image
FiringFrequency = []  # to input_count number of spikes each neuron emits during an epoch

# List of sample video files
# train video
train_raw = [
  '/data/SNN/train/magno_dataset/uva_train_01.avi'
  ...]
train_label = [
  '/data/SNN/test/magno_dataset/magno_train_01.avi'
  ...]
# test video
test_raw = [
  '/data/SNN/test/magno_dataset/uva_test_01.avi'
  ...]
test_label = [
  '/data/SNN/test/magno_dataset/magno_test_01.avi'
  ...]


traindataset = UvaDataset(train_raw, train_label)
testdataset =UvaDataset(test_raw, test_label)

# Create DataLoader
traindata_loader = DataLoader(traindataset, batch_size=1, shuffle=True)
images, labels = next(iter(traindata_loader))  # image-->(1,2,100,120) labels-->（1，100，120）
images = cp.asarray(images[0,:,:,:])
labels = cp.asarray(labels)
testdata_loader = DataLoader(testdataset, batch_size=1, shuffle=False)

# Building the model
layerSize = [[images[0].shape[0], images[0].shape[1]], [NumOfClasses, 1]]
x = cp.mgrid[0:layerSize[0][0], 0:layerSize[0][1]]  
SpikeImage = cp.zeros((layerSize[0][0], layerSize[0][1], tmax + 1))  

# To set the dynamic threshold
baseThr = 0  # base threshold 
de_factor = 0.5  # threshold learning rate
SpikeList = [SpikeImage] + Spikes  # first data [SpikeImage] + network before Spikes
DynamicThr = np.full((NumOfClasses,tmax + 1), baseThr)

# Initializing the network
np.random.seed(0)
for layer in range(Nlayers):
    W.append(cp.asarray(
        (b[layer] - a[layer]) * np.random.random_sample((Nnrn[layer], layerSize[layer][0], layerSize[layer][1])) + a[
            layer])) 
    firingTime.append(cp.asarray(np.zeros(Nnrn[layer])))
    Spikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1))))  # (12000, 1, 257)
    # TargetSpikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1))))  # (12000, 1, 257)
    X.append(cp.asarray(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]]))

if loading:
    W = np.load(LoadFrom, allow_pickle=True)
SpikeList = [SpikeImage] + Spikes  # Initial data [SpikeImage] + data Spikes after passing through the network layer

# save output
output_dir = '/pony/.../'
filename = '...'

test_MAE = 0
for i, (raw_frames, label_frame) in enumerate(testdata_loader):
    test_images = cp.asarray(raw_frames[0, :, :, :])
    labels = cp.asarray(label_frame)
    for iteration in range(len(test_images)):
        SpikeImage[:, :, :] = 0
        SpikeImage[x[0], x[1], test_images[iteration]] = 1
        SpikeList = [SpikeImage] + Spikes
        DynamicThr = np.full((NumOfClasses, tmax + 1), baseThr)
        for layer in range(Nlayers):
            Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
            for n in range(NumOfClasses):
                spk_time_n = test_images[iteration].flatten()
                spk_time = spk_time_n[n]
                Thr = baseThr + spk_time * de_factor
                DynamicThr[n, :] = Thr
            Voltage[:, tmax] = DynamicThr[:, 0] + 1
            firingTime[layer] = cp.argmax(Voltage > DynamicThr, axis=1).astype(float) + 1
            firingTime[layer][firingTime[layer] > tmax] = tmax
            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
            # TargetSpikes[layer][X[layer][0], X[layer][1], firingTarget.reshape(Nnrn[layer], 1).astype(int)] = 1
        firingTarget = labels.flatten()
        f = firingTime[0].reshape(100, 120)
        # save_gray_image(test_images[0, :, :], filename=filename,output_dir=output_dir1, index=i)
        save_gray_image(f,filename=filename,output_dir=output_dir, index=i) # Save test set output image
        # print("save")
