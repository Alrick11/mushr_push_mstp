#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:35:59 2020

@author: alrick
"""
#Model Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

class PosModel(nn.Module):
    """
    LSTM Model (Autoregressive)
    """
    
    def __init__(self, x_enc_feat, u_enc_feat, dropout=0.0, n_layers=2, rel=False):
        """
        Initializing layers.
        Model architecture: can be visualized in the model.onxx file.
        """
        
        super(PosModel, self).__init__()
        self.bpos_enc = nn.Sequential(nn.Linear(4, x_enc_feat//2), nn.ReLU(),\
                                      nn.Linear(x_enc_feat//2, x_enc_feat//2), nn.ReLU(),\
                                      nn.Linear(x_enc_feat//2, x_enc_feat))
        
        if not rel:
            self.cpos_enc = nn.Sequential(nn.Linear(4, x_enc_feat//2), nn.ReLU(),\
                                          nn.Linear(x_enc_feat//2, x_enc_feat))
        
        self.rel=rel
        self.u_enc = nn.Sequential(nn.Linear(2, u_enc_feat//2), nn.ReLU(),\
                                   nn.Linear(u_enc_feat//2, u_enc_feat//2), nn.ReLU(),\
                                   nn.Linear(u_enc_feat//2, u_enc_feat))
        
        self.lstm = nn.LSTM(input_size=x_enc_feat+u_enc_feat, \
                            hidden_size=x_enc_feat+u_enc_feat, \
                                num_layers=n_layers,\
                                    batch_first=True,\
                                        dropout=dropout)
            
        self.x = nn.Sequential(nn.Linear(x_enc_feat+u_enc_feat, 100),\
                                     nn.ReLU(), \
                                         nn.Linear(100, 1))
            
        self.y = nn.Sequential(nn.Linear(x_enc_feat+u_enc_feat+1, 100),\
                                     nn.ReLU(), \
                                         nn.Linear(100, 1))
            
        self.theta = nn.Sequential(nn.Linear(x_enc_feat+u_enc_feat+1, 100),\
                                     nn.ReLU(), \
                                         nn.Linear(100, 1))
        
    def forward(self, x_pos_b, x_pos_c, u, hidden_state):
        """
        Forward loop, when self.rel is True (relative dataset),
        then we dont need an encoder layer for car pose.
        """
        if not self.rel:
            x = torch.cat((self.bpos_enc(x_pos_b)*self.cpos_enc(x_pos_c), self.u_enc(u)), dim=2)
        else:
            x = torch.cat((self.bpos_enc(x_pos_b), self.u_enc(u)), dim=2)
        x, hidden_state = self.lstm(x, hidden_state)
        x_ac = self.x(x)
        y = self.y(torch.cat((x, x_ac), dim=2))
        
        return x_ac, y, self.theta(torch.cat((x,y), dim=2)), hidden_state
        
    def loss(self, preds, labels):
        '''
        Any regression loss function will work.
        preds shape:
        labels shape: (NxSx4)
        '''
        return F.smooth_l1_loss(preds, labels, reduction='mean')

class SimpleRegression(nn.Module):
    """
    Simple regression model
    """
    
    def __init__(self, rel=False):
        """
        Initializing layers. This model is for comparison against the LSTM model
        """
        super(SimpleRegression, self).__init__()
        self.x = nn.Sequential(nn.Linear(6, 100),\
                                     nn.ReLU(), \
                                         nn.Linear(100, 1))
            
        self.y = nn.Sequential(nn.Linear(6, 100),\
                                     nn.ReLU(), \
                                         nn.Linear(100, 1))
            
        self.theta = nn.Sequential(nn.Linear(6, 100),\
                                     nn.ReLU(), \
                                         nn.Linear(100, 1))
        
    def forward(self, x_pos_b, x_pos_c, u, hidden_state):
        """
        Forward loop
        """
        return self.x(torch.cat((x_pos_b,u), dim=2)),\
            self.y(torch.cat((x_pos_b,u), dim=2)), \
                self.theta(torch.cat((x_pos_b,u), dim=2)),\
                    hidden_state
        
    def loss(self, preds, labels):
        '''
        preds shape:
        labels shape: (NxSx4)
        '''
        return F.mse_loss(preds, labels, reduction='mean')

if __name__=="__main__":
    """
    Code testing.
    """
    device=torch.device('cuda:0')
    print ("Device: ", device)
    model=PosModel(100,50,0).to(device)
    model.train()
    
    pos_b=torch.rand(25,5,4).to(device)
    pos_c=torch.rand(25,5,4).to(device)
    pos_u=torch.rand(25,5,2).to(device)
    hidden=(torch.zeros(2,25,150).to(device), torch.zeros(2,25,150).to(device))
    labels=torch.rand(25,5,3).to(device)
    
    x,y,t, _=model(pos_b, pos_c, pos_u, hidden)
    print ("Out shapes: ", x.shape,y.shape, t.shape)
    loss = model.loss(torch.cat((x,y,t),dim=2), labels)
    print ("Loss: ", loss.item())
        