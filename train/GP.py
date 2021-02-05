#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:34:43 2020

@author: alrick
"""
#SkLearn Gaussian Process
import numpy as np
import torch
import pickle
from copy import copy

from sklearn.utils.random import sample_without_replacement
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

from DataLoader import BlockPosDataLoader

#Initialize parameters
train_file='~/hard_data/Alrick/train_data/train_f.csv.gz'
test_seen_file='~/hard_data/Alrick/train_data/test_seen_f.csv.gz'
test_unseen_file='~/hard_data/Alrick/train_data/test_unseen_f.csv.gz'
gp_train_size=2000
batch=1

#Initialize the DataLoaders
train_loader = BlockPosDataLoader(train_file,1, 1,rel=True)
test_seen_loader = BlockPosDataLoader(test_seen_file, 1, 1, rel=True)
test_unseen_loader = BlockPosDataLoader(test_unseen_file, 1, 1, rel=True)

print ("Lens: ", len(train_loader), len(test_seen_loader), len(test_unseen_loader),"\n")

min_seen_mse = 10000000000

"""
Train GP on subset of the dataset, and choose the GP that performs the best.
"""

for k in range(15):
    print ("\nTrial %d...\n"%(k))
    train_ids = sample_without_replacement(len(train_loader), gp_train_size)
    
    data = np.array([torch.cat((train_loader[i][0].squeeze()[:3], train_loader[i][2].squeeze()), dim=0).numpy() for i in train_ids])
    labels=np.array([train_loader[i][3].squeeze().numpy() for i in train_ids])
    
    data_test_seen = np.array([torch.cat((test_seen_loader[i][0].squeeze()[:3], \
                                     test_seen_loader[i][2].squeeze()), dim=0).numpy() \
                          for i in range(len(test_seen_loader))])
    labels_test_seen =np.array([test_seen_loader[i][3].squeeze().numpy() for i in range(len(test_seen_loader))])
    
    data_test_unseen = np.array([torch.cat((test_unseen_loader[i][0].squeeze()[:3], \
                                     test_unseen_loader[i][2].squeeze()), dim=0).numpy() \
                          for i in range(len(test_unseen_loader))])
    labels_test_unseen =np.array([test_unseen_loader[i][3].squeeze().numpy() for i in range(len(test_unseen_loader))])
    
    #Define kernel and model
    print ("Training ...\n")
    kernel = C(1.0) * RBF(length_scale=np.ones(data.shape[1])) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-05, 10.0))
    model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(data, labels)
    
    #Print the score
    print ("R^2 value (Score): ", model.score(data, labels))
    
    #Evaluate the model
    preds_seen = model.predict(data_test_seen)
    preds_unseen = model.predict(data_test_unseen)
    
    #Get the MSE values
    seen_mse = (np.sqrt((preds_seen-labels_test_seen)**2).sum(1)).mean(0)
    unseen_mse = (np.sqrt((preds_unseen-labels_test_unseen)**2).sum(1)).mean(0)
    
    if seen_mse < min_seen_mse:
        best_model = copy(model)
        min_seen_mse = seen_mse
    
    #Get mean squared error
    print ("(Seen) Mean squared error:", seen_mse)
    print ("(Unseen) Mean squared error:", unseen_mse)
    
filename = "gp.pickle"

print ("Saving best model")
pickle.dump(best_model, open(filename, 'wb'))
