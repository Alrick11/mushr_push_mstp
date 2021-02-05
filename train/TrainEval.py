#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:36:14 2020

@author: alrick
"""
#Training and Evaluation
import os
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm.auto import tqdm

from ModelArchitecture import PosModel, SimpleRegression
import utils

_PLT_TYPES=['loss', 'table']

def Write_Tensorboard(writer, train_loss=None, test_loss=None, test_u_loss=None, \
                      plt_type=None, embs=None, preds_s=None, labels_s=None, \
                          preds_u=None, labels_u=None, train_start=0, test_start=0):
    """
    Tensorboard visualization. Saving plots and tables to Tensorboard.
    """
    
    if plt_type not in _PLT_TYPES:
        raise Exception ("Invalid plt type %s.\nChoose from %s"%(plt_type, ' '.join(_PLT_TYPES)))
    
    if plt_type=='loss':
        if train_loss!=None:
            for i, loss in enumerate(train_loss):
                writer.add_scalar('Loss/Train loss', loss, i+train_start)
        
        if test_loss!=None:
            for i, loss in enumerate(test_loss):
                writer.add_scalar('Loss/Test loss', loss, i+test_start)
                
        if test_u_loss!=None:
            for i, loss in enumerate(test_u_loss):
                writer.add_scalar('Loss/Test Unseen loss', loss, i)
    
    elif plt_type=='table':
        if preds_s!=None and labels_s!=None:

            writer.add_hparams({'x_p':preds_s[0].item(), 'y_p':preds_s[1].item(), 't_p':preds_s[2].item(), \
                                'x_l':labels_s[0].item(), 'y_l':labels_s[1].item(), 't_l':labels_s[2].item()}, \
                               {'hparam/x_d':torch.abs(preds_s[0]-labels_s[0]), 'hparam/y_d':torch.abs(preds_s[1]-labels_s[1]),\
                                'hparam/t_d':torch.abs(preds_s[2]-labels_s[2]), '\
                                hparam/norm':sum((a-b)**2 for a,b in zip(preds_s.tolist(), labels_s.tolist()))**0.5})
                    
        if preds_u!=None and labels_u!=None:
            for i, (pred, label) in enumerate(zip(preds_u, labels_u)):
                writer.add_hparams({'x_p':pred[0], 'y_p':pred[1], 't_p':pred[2], \
                                    'x_l':label[0], 'y_l':label[1], 't_l':label[2]}, \
                                   {'hparam/x_d':abs(pred[0]-label[0]), 'hparam/y_d':abs(pred[1]-label[1]),\
                                    'hparam/t_d':abs(pred[2]-label[2]), 'hparam/norm':sum((a-b)**2 for a,b in zip(pred, label))**0.5},\
                                       run_name='Unseen params')
                    
def evaluate(writer, model, test_loader, device, hp, total=10, write_table=False, vis_traj=False, prefix=''):
    """
    Evaluated the model using test loaders and trained model
    """
    model.eval()
    
    prediction = [[] for _ in range(hp['batch_size'])]
    t_loss = [[] for _ in range(hp['batch_size'])]
    traj_state = [[] for _ in range(hp['batch_size'])]
    t_loss_avg=[]
    
    indices=[]
    n=len(test_loader)
    
    count=0
    while len(indices)<total and count<total*5:
        val=random.randint(0,n)
        count+=1
        if val not in indices:
            indices.append(val)
    
    if vis_traj:
        traj=[[] for _ in range(hp['batch_size'])]
    
    print ("Testing in progress...")
    with torch.no_grad():
        hidden_state=(torch.zeros(hp['n_layers'],hp['batch_size'],\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device), \
                      torch.zeros(hp['n_layers'],hp['batch_size'],\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device))
        
        for i in range(len(test_loader)):
            (data_b, data_c, data_u, labels_p) = test_loader[i]
            
            data_b, data_c, data_u, labels_p = data_b.to(device), data_c.to(device),\
                data_u.to(device), labels_p.to(device)
                
            x,y,theta,hidden_state = model(data_b, data_c, data_u, hidden_state)
            
            preds=torch.cat((x,y,theta),dim=2)

            #Saving mse losses.
            t_loss_avg.append(torch.sqrt((preds-labels_p).pow(2).sum(-1)).mean().item())
            
            for j in range(hp['batch_size']):
                prediction[j].append(preds[j].cpu())
                t_loss[j].append((preds[j]-labels_p[j]).cpu())
                traj_state[j].append(data_b[j, :, 3].cpu())
            
            if vis_traj:
                #Saving trajectories to Visualize
                for j in range(hp['batch_size']):
                    if hp['relative']:
                        traj[j].append((x[j].cpu().squeeze()+data_c[j,:,0].cpu(), \
                                        y[j].cpu().squeeze()+data_c[j,:,1].cpu(), \
                                        labels_p[j].cpu()+data_c[j,:,:3].cpu(), \
                                        data_b[j].cpu()))
                    else:
                        traj[j].append((x[j].cpu(), y[j].cpu(), \
                                        labels_p[j].cpu(), data_b[j].cpu()))
            
            if i in indices and write_table:
                #Create a table in tensoboard to check the values
                for j in range(min(100, hp['batch_size'])):
                    Write_Tensorboard(writer, \
                    preds_s=torch.cat((x[j,0,:],y[j,0,:],theta[j,0,:]), dim=0),\
                        labels_s=labels_p[j,0,:], plt_type='table')
                        
    if vis_traj:
        utils.Visualize_traj(traj, hp, prefix=prefix)
    
    return t_loss_avg, t_loss, traj_state, prediction


def evaluate_multi_step(writer, model, test_loader, device, hp, total=10, write_table=False, vis_traj=False, prefix=''):
    """
    Evaluated the model using test loaders and trained model, 
    for multi-step that is use only first data point of every trajectory
    of the test data.
    Only works in relative case.
    """
    model.eval()
    
    prediction = [[] for _ in range(hp['batch_size'])]
    t_loss = [[] for _ in range(hp['batch_size'])]
    traj_state = [[] for _ in range(hp['batch_size'])]
    t_loss_avg=[]
    
    indices=[]
    n=len(test_loader)
    
    count=0
    while len(indices)<total and count<total*5:
        val=random.randint(0,n)
        count+=1
        if val not in indices:
            indices.append(val)
    
    if vis_traj:
        traj=[[] for _ in range(hp['batch_size'])]
    
    print ("Testing in progress...")
    with torch.no_grad():
        hidden_state=(torch.zeros(hp['n_layers'],hp['batch_size'],\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device), \
                      torch.zeros(hp['n_layers'],hp['batch_size'],\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device))
        
        for i in range(len(test_loader)):
            (data_b, data_c, data_u, labels_p) = test_loader[i]
            
            data_b, data_c, data_u, labels_p = data_b.to(device), data_c.to(device),\
                data_u.to(device), labels_p.to(device)
            
            if i==0:
                x,y,theta,hidden_state = model(data_b, data_c, data_u, hidden_state)
            else:
                x,y,theta,hidden_state = model(utils.replace_traj_start(x,y,theta,data_b).to(device), data_c, data_u, hidden_state)
            
            preds=torch.cat((x,y,theta),dim=2)
            
            #Saving mse loss.
            t_loss_avg.append(torch.sqrt((preds-labels_p).pow(2).sum(-1)).mean().item())
            
            for j in range(hp['batch_size']):
                prediction[j].append(preds[j].cpu())
                t_loss[j].append((preds[j]-labels_p[j]).cpu())
                traj_state[j].append(data_b[j, :, 3].cpu())
            
            if vis_traj:
                #Saving trajectories to Visualize
                for j in range(hp['batch_size']):
                    if hp['relative']:
                        traj[j].append((x[j].cpu().squeeze()+data_c[j,:,0].cpu(), \
                                        y[j].cpu().squeeze()+data_c[j,:,1].cpu(), \
                                        labels_p[j].cpu()+data_c[j,:,:3].cpu(), \
                                        data_b[j].cpu()))
                    else:
                        traj[j].append((x[j].cpu(), y[j].cpu(), \
                                        labels_p[j].cpu(), data_b[j].cpu()))
            
            if i in indices and write_table:
                #Create a table in tensoboard to check the values
                for j in range(min(100, hp['batch_size'])):
                    Write_Tensorboard(writer, \
                    preds_s=torch.cat((x[j,0,:],y[j,0,:],theta[j,0,:]), dim=0),\
                        labels_s=labels_p[j,0,:], plt_type='table')
                        
    if vis_traj:
        utils.Visualize_traj(traj, hp, prefix=prefix)
    
    return t_loss_avg, t_loss, traj_state, prediction


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(train_loader, test_seen_loader, test_unseen_loader, **hp):
    """
    Training model.
    """
    
    device=torch.device(hp['device'])
    
    #Initialize the model
    if hp['is_lstm']:
        print ("Loading LSTM Model...\n")
        model = PosModel(hp['x_enc_feat'], hp['u_enc_feat'], hp['dropout'], \
                          n_layers=hp['n_layers'], rel=hp['relative']).to(device)
    else:
        print ("Loading Regression Model...\n")
        model = SimpleRegression().to(device)
    model.train()
    
    #Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    
    #Initialize the Tensorboard summaryWriter the writer
    writer=SummaryWriter(hp['writer_loc'])
    
    #Initialize Lists
    train_loss, test_loss, train_count, test_count=[], [], 0, 0
    train_start, test_start=0,0
    
    #Execute training loop
    for epoch in range(hp['epochs']+1):
        
        hidden_state=(torch.zeros(hp['n_layers'],hp['batch_size'],\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device), \
                      torch.zeros(hp['n_layers'],hp['batch_size'],\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device))
        
        #Reduce learning rate from suggested epoch onwards
        if epoch==90:
            for g in optimizer.param_groups:
                g['lr']=1e-5
        
        for i in tqdm(range(len(train_loader)), desc="Epoch %d"%epoch, position=0, leave=True):
            (data_b, data_c, data_u, labels_p)=train_loader[i]
            
            optimizer.zero_grad()
            
            data_b, data_c, data_u, labels_p = data_b.to(device), data_c.to(device),\
                data_u.to(device), labels_p.to(device)
            
            x,y,t,hidden_state = model(data_b, data_c, data_u, hidden_state)
            
            loss=model.loss(torch.cat((x,y,t),dim=2), labels_p)
            
            hidden_state = repackage_hidden(hidden_state)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
            #Evaluate in "interval" steps
            if train_count%hp['interval']==0:
                t_loss, _, _, _= evaluate(writer, model, test_seen_loader, device, hp)
                test_loss+=[sum(t_loss)/len(t_loss)]
                test_count+=len(test_seen_loader)
                model.train()
                
            train_count+=1
        
        #Write to summary writer
        if (epoch+1)%hp['write_interval']==0 or epoch==hp['epochs']:
            Write_Tensorboard(writer, train_loss=train_loss, test_loss=test_loss, \
                              plt_type='loss', train_start=train_start, \
                                  test_start=test_start)
                
            train_start+=len(train_loss)
            test_start+=len(test_loss)
            train_loss, test_loss=[],[]
        
    #Finally evaluate on unseen tests
    #Visualize only multi-step trajectories. Can visualize only one at a time.
    #Since they are being saved at the same location with the same name.
    #Clear Trajectory addr
    os.system("rm -rf %s/*"%(hp['traj_save_addr']))
    
    avg_loss_ts, loss_ts, traj_state_ts, preds_ts = \
        evaluate(writer, model, test_seen_loader, device, hp, \
                 write_table=False, vis_traj=True, prefix='ts')
    avg_loss_ms, loss_ms, traj_state_ms, preds_ms = \
        evaluate_multi_step(writer, model, test_seen_loader, device, hp, \
                            write_table=True, vis_traj=True, prefix='ms')
    avg_loss_us, loss_us, traj_state_us, preds_us = \
        evaluate(writer, model, test_unseen_loader, device, hp, \
                 write_table=False)
    
    #Initialize hidden state for adding model graph to Tensorboard file.
    hidden_state=(torch.zeros(hp['n_layers'],1,\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device), \
                      torch.zeros(hp['n_layers'],1,\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device))
    
    #Add Model to Tensorboard    
    writer.add_graph(model, [data_b[0].unsqueeze(0),data_c[0].unsqueeze(0),data_u[0].unsqueeze(0),hidden_state], verbose=False)
    
    writer.close()
    
    hidden_state=(torch.zeros(hp['n_layers'],hp['batch_size'],\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device), \
                      torch.zeros(hp['n_layers'],hp['batch_size'],\
                                  hp['x_enc_feat']+hp['u_enc_feat']).to(device))
    
    #Saving model
    if hp['is_lstm']:
        torch.save(model.state_dict(), os.path.join(hp['model_path'], 'model_lstm.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(hp['model_path'], 'model_sr.pt'))
        
    #Get Model architecture using Netron package .onnx format
    print ("Getting Model architecture...\n")
    torch.onnx.export(model, (data_b, data_c, data_u, hidden_state), "model.onnx", verbose=True,\
                          input_names=['data_b','data_c','data_u','hidden_state'],\
                          output_names=['x','y','theta','hidden_state'])
    
    #Metrics for Multi-step prediction...
    name=''
    if hp['is_lstm']:name='lstm'
    else:name='sr'
    
    #Generating log files
    traj_x, traj_y, traj_theta = utils._split_into_Traj(loss_ts, traj_state_ts, hp)
    utils.write_trajs([traj_x, traj_y, traj_theta], 'log_%s_ts.json'%(name))
    
    traj_x, traj_y, traj_theta = utils._split_into_Traj(loss_ms, traj_state_ms, hp)
    utils.write_trajs([traj_x, traj_y, traj_theta], 'log_%s_ms.json'%(name))
    
    print ("Test Multi_step trajectory MSE:\n", utils.get_metrics('log_sr_ms.json', model_name=name+'_ms'), '\n')
    
    #Report Final test performance.
    print ("\nSeen loss performance: ", sum(avg_loss_ts)/len(avg_loss_ts))
    print ("\nSeen Multi-Step loss performance: ", sum(avg_loss_ms)/len(avg_loss_ms))
    print ("\nUnseen loss performance: ", sum(avg_loss_us)/len(avg_loss_us))