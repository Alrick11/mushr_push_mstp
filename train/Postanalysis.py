#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:24:31 2021

@author: alrick
"""
#Running plotting and other things with model
import os
import torch
import utils
from torch.utils.tensorboard import SummaryWriter

from TrainEval import evaluate, evaluate_multi_step
from DataLoader import BlockPosDataLoader
from ModelArchitecture import PosModel, SimpleRegression

def Generate_metrics(vis_traj=False, **hp):

    if not all(elem in os.listdir() for elem in ['log_lstm_ts.json', 'log_lstm_ms.json',\
                                                 'log_sr_ts.json', 'log_sr_ms.json'])\
        or vis_traj:
        
        test_seen_loader = BlockPosDataLoader(hp['test_seen_file'], \
                                              hp['sequence_len'], \
                                                  hp['batch_size'], \
                                                      rel=hp['relative'])
        
        #Get model paths
        lstm_path = os.path.join(hp['model_path'], 'model_lstm.pt')
        sr_path = os.path.join(hp['model_path'], 'model_sr.pt')
        device=torch.device(hp['device'])
        
        #Load the model
        model_lstm = PosModel(hp['x_enc_feat'], hp['u_enc_feat'], hp['dropout'], \
                              n_layers=hp['n_layers'], rel=hp['relative']).to(device)
        model_lstm.load_state_dict(torch.load(lstm_path))
        
        model_sr = SimpleRegression().to(device)
        model_sr.load_state_dict(torch.load(sr_path))
        
        writer=None
        
        #Run test
        
        #Running normal testing
        avg_loss_ts_lstm, loss_ts_lstm, traj_state_ts_lstm, preds_ts_lstm = \
            evaluate(writer, model_lstm, test_seen_loader, device, hp, \
                     write_table=False, vis_traj=True, prefix='ts_lstm')
                
        avg_loss_ts_sr, loss_ts_sr, traj_state_ts_sr, preds_ts_sr = \
            evaluate(writer, model_sr, test_seen_loader, device, hp, \
                     write_table=False, vis_traj=True, prefix='ts_sr')
        
        #Running multi-step prediction
        avg_loss_ms_lstm, loss_ms_lstm, traj_state_ms_lstm, preds_ms_lstm = \
            evaluate_multi_step(writer, model_lstm, test_seen_loader, device, hp, \
                                write_table=False, vis_traj=True, prefix='ms_lstm')
                
        avg_loss_ms_sr, loss_ms_sr, traj_state_ms_sr, preds_ms_sr = \
            evaluate_multi_step(writer, model_sr, test_seen_loader, device, hp, \
                                write_table=False, vis_traj=True, prefix='ms_SR')
        
        #Split Trajs first
        #LSTM
        traj_x, traj_y, traj_theta = utils._split_into_Traj(loss_ts_lstm, traj_state_ts_lstm, hp)
        utils.write_trajs([traj_x, traj_y, traj_theta], 'log_lstm_ts.json')
        
        traj_x, traj_y, traj_theta = utils._split_into_Traj(loss_ms_lstm, traj_state_ms_lstm, hp)
        utils.write_trajs([traj_x, traj_y, traj_theta], 'log_lstm_ms.json')
        
        #SR
        traj_x, traj_y, traj_theta = utils._split_into_Traj(loss_ts_sr, traj_state_ts_sr, hp)
        utils.write_trajs([traj_x, traj_y, traj_theta], 'log_sr_ts.json')
        
        traj_x, traj_y, traj_theta = utils._split_into_Traj(loss_ms_sr, traj_state_ms_sr, hp)
        utils.write_trajs([traj_x, traj_y, traj_theta], 'log_sr_ms.json')
        
        #Print Average metrics
        print ("\n:::Test performance of LSTM:::")
        print ("\nSeen loss performance: ", sum(avg_loss_ts_lstm)/len(avg_loss_ts_lstm))
        print ("\nSeen Multi-Step loss performance: ", sum(avg_loss_ms_lstm)/len(avg_loss_ms_lstm))
        
        print ("\n:::Test performance of SR:::")
        print ("\nSeen loss performance: ", sum(avg_loss_ts_sr)/len(avg_loss_ts_sr))
        print ("\nSeen Multi-Step loss performance: ", sum(avg_loss_ms_sr)/len(avg_loss_ms_sr))
        
            
    #Evaluate the metrics
    print ("Test Multi_step trajectory MSE LSTM:\n")
    utils.get_metrics('log_lstm_ms.json', model_name='lstm_ms')
    
    print ("Testing Multi_step trajectory MSE Simple Regression:\n")
    utils.get_metrics('log_sr_ms.json', model_name='sr_ms')
    