#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:14:19 2020

@author: alrick
"""
#Visualize paths/Model performance
import os
import torch
import matplotlib.pyplot as plt
from DataLoader import BlockPosDataLoader
import numpy as np
from pandas import read_csv
from ast import literal_eval
import seaborn as sns
import math
import json
from pandas import DataFrame
from collections import defaultdict

def replace_traj_start(x,y,theta,data_b):
    """
    Replace the start of the trajectory with data_b values. 
    Used for testing model performance for Multi-step trajectory prediction.
    """
    batch, seq = data_b.shape[0], data_b.shape[1]
    
    result = torch.zeros_like(data_b)
    
    for i in range(batch):
        for j in range(seq):
            if data_b[i,j,3]==1:
                result[i,j,:]=data_b[i,j,:]
            else:
                result[i,j,0]=x[i,j,0]
                result[i,j,1]=y[i,j,0]
                result[i,j,2]=theta[i,j,0]
                result[i,j,3]=data_b[i,j,3]
                
    return result
            

def Visualize_traj(traj, hp, prefix=''):
    """
    Visualizing predicted and target (actual) trajectories
    """
    count=0
    save_addr=hp['traj_save_addr']
    print ("\nVisualizing traj ...\n")
    
    if not os.path.exists(save_addr):
        os.mkdir(save_addr)
        
    if not os.path.exists(os.path.join(save_addr, prefix)):
        os.mkdir(os.path.join(save_addr, prefix))
    
    markersize, markeredgewidth = 0.5, 1
    x,y,label_x,label_y=[],[],[],[]
    freq = defaultdict()
    for i in range(hp['batch_size']):
        t=traj[i]
        
        #Combine trajs between 1s, the fourth dimension of data_b
        #1 in the fourth dimesion indicates the start of a new trajectory.
        for j in range(len(t)):
            for k in range(hp['sequence_len']):
                if x==[]:
                    if t[j][3][k,-1]==1:
                        x.append(t[j][0][k])
                        y.append(t[j][1][k])
                        label_x.append(t[j][2][k,0])
                        label_y.append(t[j][2][k,1])
                    else:
                        continue
                    
                else:
                    if t[j][3][k][-1]!=1:
                        x.append(t[j][0][k])
                        y.append(t[j][1][k])
                        label_x.append(t[j][2][k,0])
                        label_y.append(t[j][2][k,1])
                    else:
                        
                        if len(x) in freq:
                            freq[len(x)]+=1
                        else:
                            freq[len(x)]=1
                            
                        plt_name = "%s_%d"%(len(x), freq[len(x)])
                        
                        fig=plt.figure()
                        plt.xlim(-15,15)
                        plt.ylim(-15,15)
                        plt.plot(torch.tensor(x).numpy(), torch.tensor(y).numpy(), 's', markersize=markersize,\
                                 markeredgewidth=markeredgewidth, markeredgecolor='r', markerfacecolor='None', alpha=0.8, label="predictions Len: %d"%(len(x)))
                        plt.plot(torch.tensor(label_x).numpy(), torch.tensor(label_y).numpy(), 's', markersize=markersize,\
                                 markeredgewidth=markeredgewidth, markeredgecolor='b', alpha=0.1, label="targets")
                        plt.xlabel('X')
                        plt.ylabel('Y')
                        plt.axes().set_aspect('equal')
                        plt.legend()
                            
                        plt.savefig("%s/%s/%s.png"%(save_addr, prefix, plt_name))

                        count+=1
                        plt.close()
                        
                        #Initialize trajs again
                        x,y,label_x,label_y=[t[j][0][k]],[t[j][1][k]],[t[j][2][k,0]],[t[j][2][k,1]]
    
    t=traj[0]
    x,y,label_x,label_y=[],[],[],[]
    for j in range(len(t)):
        for k in range(hp['sequence_len']):
                x.append(t[j][0][k])
                y.append(t[j][1][k])
                label_x.append(t[j][2][k,0])
                label_y.append(t[j][2][k,1])

    fig=plt.figure()
    
    if len(x) in freq:
        freq[len(x)]+=1
    else:
        freq[len(x)]=1
        
    plt_name = "%s_%d"%(len(x), freq[len(x)])
    
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.plot(torch.tensor(x).numpy(), torch.tensor(y).numpy(), 's', markersize=markersize,\
             markeredgewidth=markeredgewidth, markeredgecolor='r', markerfacecolor='None', alpha=0.8, \
                 label="predictions Len: %d"%(len(x)))
    plt.plot(torch.tensor(label_x).numpy(), torch.tensor(label_y).numpy(), 's', markersize=markersize,\
             markeredgewidth=markeredgewidth, markeredgecolor='b', alpha=0.1, label="targets")
    plt.savefig("%s/%s/%s.png"%(save_addr, prefix, plt_name))
    
    plt.close()
    
    #Initialize trajs again
    x,y,label_x,label_y=[],[],[],[]
    
    print ("\nDone visualizing trajectories.")
            
            
def test_dataLoader(dataloader, batch_size):
    """
    Printing DataLoader values. Created for testing the Dataloader
    """
    for i, (data_b, data_c, data_u, labels_p) in enumerate(dataloader):
        print (data_u[0,:])

def _plot_data(x,y,name, l=''):
    """
    Basic plotting function
    """
    sns.set_style('whitegrid')
    fig=plt.figure()
    # plt.xlim(-5,5)
    # plt.ylim(-5,5)
    if x:
        plt.plot(x,y,'b+', label=l)
    else:
        plt.scatter(np.arange(y.shape[0]), y, s=30, alpha=0.5, label=l)
        plt.plot(np.arange(y.shape[0]), y)
    plt.ylabel('Average MSE across all trajectories')
    plt.xlabel('Index of point in trajectory')
    plt.legend()
    plt.savefig(name)
    plt.close()

def _plot_sns(data, name, label=''):
    """
    Plotting average trajectory error with 95% confidence interval
    """
    sns.set_style('darkgrid')
    fig = plt.figure()
    # sns_plot = sns.lineplot(data=data, x='Index', y='Absolute Error', \
    #                         legend='brief', label=label)
    sns_plot = sns.lineplot(data=data, label=label)
    
    plt.savefig(name)
    plt.close()

def dataLoader_traj(batch, dataloader, batch_size, rel=False):
    """
    Testing Dataloader. Can ignore this function.
    """
    
    if batch>=batch_size:
        raise Exception("Invalid batch size")
    
    seq_len=dataloader.seq_len
    x,y=[],[]
    start=True
    count=0
    
    for b in range(batch_size):
        for i, (data_b, data_c, _,_) in enumerate(dataloader):
            data_b, data_c = data_b[b,:], data_c[b,:]
            for j in range(seq_len):
                if start:
                    if rel:
                        if data_b[j,-1]==1:
                            x.append((data_b[j,0]+data_c[j,0]).item())
                            y.append((data_b[j,1]+data_c[j,1]).item())
                            start=False
                            continue
                    else:
                        if data_b[j,-1]==1:
                            x.append(data_b[j,0])
                            y.append(data_b[j,1])
                            start=False
                            continue
                else:
                    if data_b[j,-1]==1:
                        start=True
                        j-=1
                        _plot_data(np.array([t.numpy() for t in x]).flatten(),\
                                  np.array([t.numpy() for t in y]).flatten(),\
                                      name="utils_vis/dfig_%d.png"%(count))
                        count+=1
                        x,y=[],[]
                                  
                        continue
                        
                    else:
                        if rel:
                            x.append((data_b[j,0]+data_c[j,0]).item())
                            y.append((data_b[j,1]+data_c[j,1]).item())
                        else:
                            x.append(data_b[j,0])
                            y.append(data_b[j,1])

def plot_ac_traj(filename):
    """
    Plotting Trajectories from Data. (Purpose of testing the dataloader)
    """
    reader=read_csv(filename)
    n=len(reader)
    i,count=0,0
    start=True
    x,y=[],[]
    while i<n:
        if start:
            pb=literal_eval(reader['prev_block_pos'][i])
            if pb[-1]==1:
                x.append(pb[0])
                y.append(pb[1])
                start=False
                
        else:
            pb=literal_eval(reader['prev_block_pos'][i])
            if pb[-1]==1:
                _plot_data(np.array(x), np.array(y), "utils_vis/ACT_%d"%(count))
                count+=1
                x,y=[],[]
                start=True
                i-=1
                
            else:
                x.append(pb[0])
                y.append(pb[1])
                
        i+=1

def _split_into_Traj(t, traj_state, hp: dict):
    """
    Split the Data into trajectories. The data has four states
    x,y,theta,state, 
    where state is 1 if the vector is the start of a trajectory,
    else it is 0.
    """
    traj_x, traj_y, traj_theta, temp=[],[],[],[]
    
    for data_t, traj_ts_t in zip(t, traj_state):
        for data, traj_ts in zip(data_t, traj_ts_t):
            for j in range(hp['sequence_len']):
                # print (traj_ts[j], data[j]
                if traj_ts[j].item()==1:
                    if temp!=[]:
                        traj_x.append([x[0].item() for x in temp])
                        traj_y.append([x[1].item() for x in temp])
                        traj_theta.append([x[2].item() for x in temp])
                        
                    temp=[]
                    temp.append(data[j,:])
                    
                else:
                    temp.append(data[j,:])
        
    if temp!=[]:
        traj_x.append([x[0].item() for x in temp])
        traj_y.append([x[1].item() for x in temp])
        traj_theta.append([x[2].item() for x in temp])

    print ("Number of trajectories: ", len(traj_x), len(traj_y), len(traj_theta))

    return traj_x, traj_y, traj_theta

def get_metrics(log_path, model_name=''):
    """
    Get error in x,y,theta over the trajectory and save it in 
    log.json file. Also plot the error over the average of all trajectories.
    """
    
    with open(log_path, 'r') as p:
        x=json.load(p)
    
    traj_x, traj_y, traj_theta = x['trajectories_x'], x['trajectories_y'], x['trajectories_theta']
    
    #Go over thetaas and rescale them
    for k in range(len(traj_theta)):
        for i in range(len(traj_theta[k])):
            while abs(traj_theta[k][i])>math.pi:
                traj_theta[k][i] = abs(2*math.pi - abs(traj_theta[k][i]))
    
    #Get averages over all trajectories and plot them
    avgs=[]
    for traj_mse, name, label in zip([traj_x, traj_y, traj_theta],\
                                     ['AbsoluteError_%s_x'%(model_name), 
                                      'AbsoluteError_%s_y'%(model_name), 
                                      'AbsoluteError_%s_theta'%(model_name)],
                                     ['x', 'y', 'theta']):

        index, flag=0, 1
        avg_traj_mse, indices=[],[]
        counts={}
        
        while flag!=0:
            flag, count, val=0,0,0
            for traj in traj_mse:
                if index < len(traj)-1:
                    # val+=traj[index]**2
                    avg_traj_mse.append(abs(traj[index]))
                    indices.append(index)
                    count+=1
                    flag=1
                else:
                    pass
                
            if flag:
                # avg_traj_mse.append(val**0.5/count)
                counts[index]=count
                index+=1
        
        #Group avg_traj and then take their averages...
        # print ("Before averaging length: ", len(avg_traj_mse))
        # avg_traj_mse = [ sum(avg_traj_mse[i:i+avg_n])/len(avg_traj_mse[i:i+avg_n]) for i in range(0,len(avg_traj_mse), avg_n) ]
        # print ("Changed len: ", len(avg_traj_mse), avg_n)
        total_counts = [counts[i] for i in indices]
        
        data = DataFrame({'Index': indices, \
                        'Absolute Error': avg_traj_mse[:], 'Count': total_counts})
        
        _plot_sns(data, name, label=label)
        # _plot_data(None, np.array(avg_traj_mse), name, l=label)
        avgs.append(avg_traj_mse[:])
        # _sns_plot_data(np.concatenate((np.arange(len(avg_traj_mse)), np.array(avg_traj_mse))), name)

    sns.set_style('darkgrid')
    n1, n2, n3 = len(avgs[0]), len(avgs[1]), len(avgs[2])
    data = DataFrame({'Index': indices[:]+ indices[:] + indices[:],\
                      'Absolute Error': avgs[0]+avgs[1]+avgs[2],\
                        'Labels': ['x' for i in range(n1)] + \
                            ['y' for i in range(n2)] + \
                                ['z' for i in range(n3)]})
        
    sns_plot = sns.lineplot(data=data, x="Index", y='Absolute Error', hue='Labels')
    fig = sns_plot.get_figure()
    fig.savefig('AbsoluteError_combined_%s'%(model_name))
    
    # fig, ax = plt.subplots(3, figsize=(10,8), sharex=True, sharey=True)
    
    # ax[0].scatter(np.arange(len(avgs[0])), np.array(avgs[0]), s=30, alpha=0.5, c='r', label='x mse - Avg: %d'%(avg_n))
    # ax[1].scatter(np.arange(len(avgs[0])), np.array(avgs[1]), s=30, alpha=0.5, c='b', label='y mse - Avg: %d'%(avg_n))
    # ax[2].scatter(np.arange(len(avgs[0])), np.array(avgs[2]), s=30, alpha=0.5, c='g', label='theta mse - Avg: %d'%(avg_n))
    
    # plt.xlabel('Index of point in Trajectory')
    # ax[1].set_ylabel('Average MSE across all trajectories')
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # plt.savefig('Average_MSE_combined_%s'%(model_name))

def write_trajs(traj, filename):
    """
    Write Trajectory into json files
    traj = [[traj_x], [traj_y], [traj_theta]]
    """
    x = {}
    x['total_traj'] = len(traj[0])
    x['smallest_traj'] = min(len(x) for x in traj[0])
    x['largest_traj'] = max(len(x) for x in traj[0])
    
    x['trajectories_x'], x['trajectories_y'], \
        x['trajectories_theta']=[],[],[]
    
    for t in traj[0]:
        for i in range(len(t)):
            if isinstance(t[i], torch.Tensor):
                t[i]=t[i].item()
        x['trajectories_x'].append(t)
        
    for t in traj[1]:
        for i in range(len(t)):
            if isinstance(t[i], torch.Tensor):
                t[i]=t[i].item()
        x['trajectories_y'].append(t)
        
    for t in traj[2]:
        for i in range(len(t)):
            if isinstance(t[i], torch.Tensor):
                t[i]=t[i].item()
        x['trajectories_theta'].append(t)
    
    with open(filename, 'w') as p:
        json.dump(x, p, indent=4, sort_keys=True)
    
if __name__=="__main__":
    """
    Testing utility functions
    """
    
    # file='/home/ugrads/hard_data/Alrick/train_data/test_seen_0.4.csv.gz'
    # relative=False
    # seq_len=2000
    # batch_size=10
    # loader=BlockPosDataLoader(file, seq_len, batch_size, rel=relative)
    # # test_dataLoader(loader, 10)
    # dataLoader_traj(0,loader,batch_size,rel=relative)
    # plot_ac_traj(file)
    
    #Testing get_metrics
    batch, seq, chunks = 500, 1, 10
    torch.manual_seed(10)
    
    t, t1, traj_state = [[] for i in range(batch)], [[] for i in range(batch)],\
                        [[] for i in range(batch)]
    
    for _ in range(chunks):
        for i in range(batch):
            t[i].append(torch.rand(seq, 3))
            t1[i].append(torch.rand(seq, 1))
            traj_state[i].append([x > 0.9 for x in t1[i]])
    
    # print ("t: \n", t[0].shape)
    # print ("\n\nTraj_state:\n", traj_state[0].shape)
    
    get_metrics(t, traj_state, {'batch_size':batch, 'sequence_len':seq})
    
    #Testing plotting preds
    # preds = [torch.rand(10,10,4) for i in range(10)]
    # plot_preds_vs_time(preds)
    
