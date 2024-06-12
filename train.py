# Imports
from __future__ import division

import math
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import raw_spike_time, magno_spike_time, calculate_dynamic_threshold, save_image
from dataloader import UvaDataset

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
loading = False  # Set it as True if you want to load a pretrained model
LoadFrom = "weights.npy"  # The pretrained model
saving = True # Set it as True if you want to save the trained model
van_best = float('inf')
train_mse_best = float('inf')
train_mae_best = float('inf')
test_mse_best = float('inf')
test_mae_best = float('inf')
Nnrn = [NumOfClasses]  # Number of neurons at hidden and output layers


# Loss
testMaeLoss = []
trainMaeLoss = []

W = []  # To hold the weights of hidden and output layers
firingTime = []  # To hold the firing times of hidden and output layers
Spikes = []  # To hold the spike trains of hidden and output layers
X = []  # To be used in converting firing times to spike trains
firingTarget = cp.zeros([NumOfClasses])  # To keep the firingTarget firing times of current image
FiringFrequency = []  # to input_count number of spikes each neuron emits during an epoch


# train video
train_raw = [...]
train_label = [...]
# test video
test_raw = [...]
test_label = [...]


# Data processing
# Directly converted to three frames a batch, if you need to modify the number of frames output at a time, please modify the dataloader.py
traindataset = UvaDataset(train_raw, train_label)
testdataset = UvaDataset(test_raw, test_label)

# DataLoader
traindata_loader = DataLoader(traindataset, batch_size=1, shuffle=True)
print("train datasets done.")
images, labels = next(iter(traindata_loader))
images = cp.asarray(images[0,:,:,:])
labels = cp.asarray(labels)

testdata_loader = DataLoader(testdataset, batch_size=1, shuffle=False)
print("test datasets done.")

# Building the model
layerSize = [[images[0].shape[0], images[0].shape[1]], [NumOfClasses, 1]]
x = cp.mgrid[0:layerSize[0][0], 0:layerSize[0][1]]  # To be used in converting raw image into a spike image
SpikeImage = cp.zeros((layerSize[0][0], layerSize[0][1], tmax + 1))  # To keep spike image 

# Initializing the network
np.random.seed(0)
for layer in range(Nlayers):
    W.append(cp.asarray(
        (b[layer] - a[layer]) * np.random.random_sample((Nnrn[layer], layerSize[layer][0], layerSize[layer][1])) + a[
            layer]))  
    firingTime.append(cp.asarray(np.zeros(Nnrn[layer])))
    Spikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1)))) 
    X.append(cp.asarray(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]]))

if loading:
    if GPU:
        W = np.load(LoadFrom, allow_pickle=True)
    else:
        for i in range(len(W)):
            W[i] = cp.asnumpy(W[i])

# To set the dynamic threshold
baseThr = 0  # base threshold 
de_factor = 0.5  # threshold learning rate
SpikeList = [SpikeImage] + Spikes  # first data [SpikeImage] + network before Spikes
DynamicThr = np.full((NumOfClasses,tmax + 1), baseThr)



# Start learning
for epoch in range(Nepoch):
    start_time = time.time()
    # Evaluating on train samples
    # Start an epoch
    for i, (raw_frames, label_frame) in enumerate(traindata_loader):
        images = cp.asarray(raw_frames[0, :, :, :])
        labels = cp.asarray(label_frame)
        for iteration in range(len(images)):
            # converting input image into spiking image
            SpikeImage[:, :, :] = 0
            SpikeImage[x[0], x[1], images[iteration]] = 1
            DynamicThr = np.full((NumOfClasses, tmax + 1), baseThr)

            # Feedforward path
            for layer in range(Nlayers):
                Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1) 
                for n in range(NumOfClasses):
                    spk_time_n = images[iteration].flatten()
                    spk_time = spk_time_n[n]
                    Thr = baseThr + spk_time * de_factor
                    DynamicThr[n,:] = Thr  
                Voltage[:, tmax] = DynamicThr[:,0] + 1  
                firingTime[layer] = cp.argmax(Voltage > DynamicThr, axis=1).astype(
                        float) + 1  
                firingTime[layer][firingTime[layer] > tmax] = tmax  
                Spikes[layer][:, :, :] = 0
                Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(
                    int)] = 1  
            firingTarget = labels.flatten()

            # Backward path
            layer = Nlayers - 1  # Output layer
            delta_o = []
            for i in range(len(firingTime[layer])):
                if (firingTarget[i] - firingTime[layer][i]) > 0:
                    delta = ((firingTarget[i] - firingTime[layer][i]) ** 2) / (tmax**2)
                    delta_o.append(delta)
                else:
                    delta = -((firingTarget[i] - firingTime[layer][i]) ** 2) / (tmax**2)
                    delta_o.append(delta)
            # grad normalization
            delta_o = cp.asarray(delta_o)
            norm = cp.linalg.norm(delta_o)
            if (norm != 0):  
                delta_o = delta_o / norm
            # Dropout
            if Dropout[layer] > 0:
                firingTime[layer][cp.asarray(np.random.permutation(Nnrn[layer])[Dropout[layer]])] = tmax
            # Updating weights
            hasFired_h = images[iteration] < firingTime[layer][:, cp.newaxis,
                                             cp.newaxis]  
            # STDP
            W[layer] -= lr[layer] * delta_o[:, cp.newaxis, cp.newaxis] * hasFired_h
            # Weight regularization
            W[layer] -= lr[layer] * lamda[layer] * W[layer]  

    # Evaluating on test samples
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
                    DynamicThr[n,:] = Thr
                Voltage[:, tmax] = DynamicThr[:,0] + 1
                firingTime[layer] = cp.argmax(Voltage > DynamicThr, axis=1).astype(float) + 1
                firingTime[layer][firingTime[layer] > tmax] = tmax
                Spikes[layer][:, :, :] = 0
                Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
            firingTarget = labels.flatten()
            mae = F.l1_loss(torch.tensor(firingTime[Nlayers - 1]), torch.tensor(firingTarget))
            test_MAE += mae
    test_MAE /= NumOfClasses
    testMaeLoss.append(test_MAE)

    print('test_epoch= ', epoch)
    print('test_MAE= ', test_MAE)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Evaluating on train samples
    # train
    train_Van = 0
    train_MAE = 0
    train_MSE = 0
    for i, (raw_frames, label_frame) in enumerate(traindata_loader):
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
                    DynamicThr[n,:] = Thr

                Voltage[:, tmax] = DynamicThr[:,0] + 1
                firingTime[layer] = cp.argmax(Voltage > DynamicThr, axis=1).astype(float) + 1
                firingTime[layer][firingTime[layer] > tmax] = tmax
                Spikes[layer][:, :, :] = 0
                Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
            firingTarget = labels.flatten()
            mae = F.l1_loss(torch.tensor(firingTime[Nlayers - 1]), torch.tensor(firingTarget))
            train_MAE += mae

    train_MAE /= NumOfClasses
    trainMaeLoss.append(train_MAE)

    print('train_epoch= ', epoch)
    print('train_MAE= ', train_MAE)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # To save the weights
    if saving:
        np.save("/data/MG-SNN/mg_snn_w/weights{}".format(epoch), W, allow_pickle=True)
        print("Dy-W done.")
        if train_MAE < train_mae_best:
            np.save("/data/MG-SNN/mg_snn_w/weights_mae_trainbest", W, allow_pickle=True)
            train_mae_best = train_MAE
        if test_MAE < test_mae_best:
            np.save("/data/MG-SNN/mg_snn_w/weights_mae_testbest", W, allow_pickle=True)
            test_mae_best = test_MAE

print("trainMaeLoss",trainMaeLoss)
print("testMaeLoss",testMaeLoss)


