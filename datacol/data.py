#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:30:08 2020

@author: alrick
"""
#Contains data essentials
import os, sys
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import compress_json
import csv, gzip
import ast
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import random
sys.path.append('/home/alrick/github_collection/mushr_push_sim/src')
import utils
from scipy.spatial.transform import Rotation

class Random_policy:
    """
    Generate random speed and steering angle
    """
    def __init__(self, speed_label, angle_label, v_low=0.3, v_high=1, \
                 a_low=-0.3, a_high=0.3, interval=2000):
        self.v_low=v_low
        self.v_high=v_high
        self.a_low=a_low
        self.a_high=a_high
        self.count=0
        self.interval=interval
        self.controls=None
        self.speed=speed_label
        self.angle=angle_label
    
    def __call__(self, state):
        return self.compute_control(state)
    
    def compute_control(self, state):
        """
        Parameters
        ----------
        state : Can be anything as it wont be used. But ideally
        np.array([block_pos, block_quat, car_pos, car_quat])
        Returns
        -------
        Values sampled from normal dist
        """
        if self.count%self.interval==0:
            self.controls = np.array([np.random.uniform(self.v_low, self.v_high), \
                        np.random.uniform(self.a_low, self.a_high)])
        
        self.count+=1
            
        return self.controls
            
class StraightPath_policy:
    """
    Outputs constant speed with 0 steering angle
    """
    def __init__(self, v):
        self.speed=v
        self.angle=0
        
    def __call__(self, state):
        return self.compute_control(state)    
        
    def compute_control(self, state):
        """
        Parameters
        ----------
        state : Same as above
        Returns
        -------
        controls to go straight
        """
        return np.array([self.speed, 0])
    
class Curved_policy:
    """
    Outputs constant speed and constant steering angle
    """
    def __init__(self, v, a):
        self.speed=v
        self.angle=a
    
    def __call__(self, state):
        return self.compute_control(state)
    
    def compute_control(self, state):
        """
        Parameters
        ----------
        state : Same as above
        Returns
        -------
        controls for going in a circle
        """
        return np.array([self.speed, self.angle])
    
def detect_contact(sim):
    """
    Detect from state whether car and block are in contact
    Returns True if they are in contact else False
    """
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        if sim.model.geom_id2name(contact.geom1)=="buddy_pusher" or sim.model.geom_id2name(contact.geom2)=="buddy_pusher":
            return True
    
    return False

def delete_common_instances(x):
    """
    Deletes data where previous block pose = next time step block pose
    """
    n = len(x['prev_block_pos'])
    indices=[]
    
    for i in range(n):
        if x['prev_block_pos'][i]==x['next_block_pos'][i]:
            indices.append(i)

    for index in sorted(indices, reverse=True):
        for key in x.keys():
            if key!='metrics':
                temp=x[key].pop(index)

                if isinstance(temp, list):
                    if temp[-1]==1:
                        x[key][index][-1]=1
                
    return x

def save_summary_writer(parent_folder):
    """
    Parameters
    ----------
    parent_folder : Destination of Tensorboard summary writer
    Returns
    -------
    None.
    """
    writer = SummaryWriter("/home/alrick/github_collection/mushr_push_sim/")
    
    with gzip.open(os.path.join(parent_folder, 'final.csv.gzip'), 'rt', newline='') as csvfile:
        reader=csv.DictReader(csvfile)
        print ("Fieldnames: ", reader.fieldnames)
        
        for i, row in enumerate(reader):
            try:
                pos = ast.literal_eval(row['prev_block_pos'])
                quat = ast.literal_eval(row['prev_block_quat'])
                writer.add_scalar('block_pos', pos[1], pos[0])
                writer.add_scalar('block_quat', quat[0], quat[2])
            except SyntaxError:
                print (row['prev_block_pos'], row['prev_block_quat'])
            
    writer.close()

def visualize_data(parent_folder, step=1):
    """
    Visualize the data
    Parameters
    ----------
    parent_folder : Search for final.json in parent folder
    Returns
    -------
    Plots
    """
    path=os.path.join(parent_folder, "final.csv.gz")
    with gzip.open(path, 'rt', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        print ("Fieldnames: ", reader.fieldnames)
        
        fig1=plt.figure(1)
        ax1, =plt.plot([],[], 'r+', label='Block Pos (x,y)')
        plt.xlim(-15,15)
        plt.ylim(-15,15)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axes().set_aspect('equal')
        plt.legend()
        
        fig2=plt.figure(2)
        ax2, =plt.plot([], 'b+', label='Block Pos (theta)')
        plt.xlim(-1000, 58000)
        plt.ylim(-200, 200)
        plt.xlabel('time')
        plt.ylabel('Theta')
        plt.axes().set_aspect('equal')
        plt.legend()
        
        try:
            for i, row in enumerate(reader):
                # print ("%d"%(i))
                if i%step!=0:
                    continue
                
                if i==0:
                    total_vals = row['total_size']
                    print ("Total vals: ", total_vals, type(total_vals))
                    pbar = tqdm(total=int(total_vals), desc="Plotting")
                    pbar.update(step)
                else:
                    pbar.update(step)

                pos = ast.literal_eval(row['prev_block_pos'])
                quat = ast.literal_eval(row['prev_block_quat'])
                ax1.set_xdata(np.append(ax1.get_xdata(), pos[0]))
                ax1.set_ydata(np.append(ax1.get_ydata(), pos[1]))
                
                ax2.set_ydata(np.append(ax2.get_ydata(), quat[2]))
                ax2.set_xdata(np.append(ax2.get_xdata(), i//step))
                plt.draw()
        except KeyboardInterrupt:
            pass
        finally:
            plt.show()

def save_json(x, name, i, v, a, path):
    """
    Parameters
    ----------
    x: dictionary (Data)
    name: name of controller
    i: Keep count of number of saves
    path: Home dir
    """
    save_path = os.path.join(path, name + "_%d_%f_%f"%(i, v, a) + ".json.gz")
    compress_json.dump(x, save_path)
    
def Create_empty_x():
    return {'prev_block_pos':[], 'prev_block_quat':[], 'next_block_pos':[],\
         'next_block_quat':[],\
            'prev_car_pos':[], 'prev_car_quat':[], 'next_car_pos':[],\
         'next_car_quat':[],\
             'speed': [], 'steering_angle':[]}

def combine_data(parent_folder):
    """
    Parameters
    ----------
    parent_folder : Folder with .json files
    Returns:
    Combined json file
    """
    lst = os.listdir(parent_folder)
    random.shuffle(lst)
    
    x = Create_empty_x()
    x['metrics']=[]
    x['total_size']=0
    total_size=0
    
    def temp_combine(k,v,idx):
        if k!='metrics':
            return v[idx]
        else:
            return v
    
    def temp_check(k, v, total_size):
        if k=='total_size':
            return total_size
        else:
            return v
    
    print ('Combining files...\n')
    with gzip.open(os.path.join(parent_folder, "final_temp.csv.gz"), 'wt+', newline='') as csvfile:
        fieldnames = list(x.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for l in lst:
            if l.endswith(".json.gz") and "policy" in l:
                p = os.path.join(parent_folder, l)
                print ("Path: ", p)
                temp_x = compress_json.load(p)
                n = len(temp_x['prev_block_pos'])
                total_size+=n
                
                for idx in range(n):
                    writer.writerow({k:temp_combine(k,v,idx) for k,v in temp_x.items()})
                
    print ("Total size: ", total_size)
        
    with gzip.open(os.path.join(parent_folder, "final_temp.csv.gz"), 'rt+', newline='') as csvfile1:
        with gzip.open(os.path.join(parent_folder, "final.csv.gz"), 'wt+', newline='') as csvfile2:
            reader=csv.DictReader(csvfile1)
            writer=csv.DictWriter(csvfile2, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(reader):
                if i==0:
                    writer.writerow({k:temp_check(k, row[k], total_size) for k in fieldnames})
                else:
                    writer.writerow({k:row[k] for k in fieldnames})
                    
    os.system('rm -f final_temp.csv.gz')
            
def split_data(parent_folder, ratio=0.7):
    """
    parent_folder : parent_folder of final.csv.gz
    Splits data into train, test, unseen tests
    """
    trainp=os.path.join(parent_folder, 'train_temp.csv.gz')
    test_seenp=os.path.join(parent_folder, 'test_seen_temp.csv.gz')
    test_unseenp=os.path.join(parent_folder, 'test_unseen_temp.csv.gz')
    finalp=os.path.join(parent_folder, 'final.csv.gz')
    
    x=Create_empty_x()
    x['total_size']=0
    fieldnames=list(x.keys())
    
    print ("Splitting Data into train, test, unseen_tests")
    
    with gzip.open(finalp, 'rt+', newline='') as p4:
        with gzip.open(trainp, 'wt+', newline='') as p1:
            with gzip.open(test_seenp, 'wt+', newline='') as p2:
                with gzip.open(test_unseenp, 'wt+', newline='') as p3:
                    train=csv.DictWriter(p1, fieldnames=fieldnames)
                    test_seen=csv.DictWriter(p2, fieldnames=fieldnames)
                    test_unseen=csv.DictWriter(p3, fieldnames=fieldnames)
                    
                    train.writeheader()
                    test_seen.writeheader()
                    test_unseen.writeheader()
                
                    final=csv.DictReader(p4)
                    
                    n_train_count=100
                    n_train, n_test_seen, n_test_unseen=0,0,0
                    cond=True
                    
                    for i, row in tqdm(enumerate(final), desc="Cond {}: ".format(cond), position=0, leave=True):
                        if i==0:
                            n_train_count=int(ratio*int(row['total_size']))
                            print ("Train count value: ", n_train_count)
                            train.writerow({k:row[k] for k in fieldnames})
                            n_train+=1
                        
                        # if i>n_train_count:print ('Prev block pos value: ', ast.literal_eval(row['prev_block_pos'])[-1])
                        if i>n_train_count and ast.literal_eval(row['prev_block_pos'])[-1]==1 and cond:
                            cond=False
                        
                        if cond:
                            if all(x < 5 for x in ast.literal_eval(row['next_block_pos'])):
                                train.writerow({k:row[k] for k in fieldnames})
                                n_train+=1
                            else:
                                test_unseen.writerow({k:row[k] for k in fieldnames})     
                                n_test_unseen+=1
                        else:
                            if all(x < 5 for x in ast.literal_eval(row['next_block_pos'])):
                                test_seen.writerow({k:row[k] for k in fieldnames})
                                n_test_seen+=1
                            else:
                                test_unseen.writerow({k:row[k] for k in fieldnames})
                                n_test_unseen+=1
    
    def temp_check(k, v, total_size):
        if k=='total_size':
            return total_size
        else:
            return v
    
    print ("Len of train: %d\nLen of test_seen: %d\nLen of test_unseen: %d"%(n_train,\
                                                n_test_seen, n_test_unseen))
    
    with gzip.open(trainp, 'rt+', newline='') as p1:
        with gzip.open(os.path.join(parent_folder, 'train.csv.gz'), 'wt+', newline='') as p2:
            reader = csv.DictReader(p1)
            writer= csv.DictWriter(p2, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(reader):
                if i==0:
                    writer.writerow({k:temp_check(k, row[k], n_train) for k in fieldnames})
                else:
                    writer.writerow({k:row[k] for k in fieldnames})
                    
    with gzip.open(test_seenp, 'rt+', newline='') as p1:
        with gzip.open(os.path.join(parent_folder, 'test_seen.csv.gz'), 'wt+', newline='') as p2:
            reader = csv.DictReader(p1)
            writer= csv.DictWriter(p2, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(reader):
                if i==0:
                    writer.writerow({k:temp_check(k, row[k], n_train) for k in fieldnames})
                else:
                    writer.writerow({k:row[k] for k in fieldnames})
                    
    with gzip.open(test_unseenp, 'rt+', newline='') as p1:
        with gzip.open(os.path.join(parent_folder, 'test_unseen.csv.gz'), 'wt+', newline='') as p2:
            reader = csv.DictReader(p1)
            writer= csv.DictWriter(p2, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(reader):
                if i==0:
                    writer.writerow({k:temp_check(k, row[k], n_train) for k in fieldnames})
                else:
                    writer.writerow({k:row[k] for k in fieldnames})
                    
    os.system('rm -f %s %s %s'%(trainp, test_seenp, test_unseenp))
    print ("Splitting Done ...")
                

def collect_data(sim, controller, runs=None, trials=None, time_limit=None, save_path=None):
    """
    Parameters
    ----------
    sim : simulator MushrPushSim
    controller : Policies defined above
    runs : (int) number of runs
    trials : (int) number of trials
    time_limit : timeout time limit for each run
    save_path : folder to save all .json files
    Returns
    -------
    None.
    """
    try:
        if runs!=None:
            sim.config['num_runs']=runs
            
        if trials!=None:
            sim.config['num_trials']=trials
            
            
        if time_limit==None:
            time_limit = 10
            
        if save_path==None:
            save_path=os.getcwd()
        
        x_count=1
        x_len_tracker=0
        
        x = Create_empty_x()
        
        while not sim.done:
            sim.setup_next_run()
            plt_count=0

            while not sim.run_done: 
                sim.setup_next_trial()

                new_exp=True
                
                start=time.time()
                while not sim.trial_done:
                    if detect_contact(sim.sim):
                        extra_dim=[1 if new_exp else 0]
                        
                        prev_state, _ = sim.get_state()
                        x['prev_block_pos'].append(np.copy(prev_state[0]).tolist()+extra_dim[:])
                        x['prev_block_quat'].append(np.copy(Rotation.from_quat(\
                                                utils.reverse_flip_quat(prev_state[1])
                                    ).as_euler('xyz', degrees=False)).tolist()+extra_dim[:])
                        
                        x['prev_car_pos'].append(np.copy(prev_state[2]).tolist()+extra_dim[:])
                        x['prev_car_quat'].append(np.copy(Rotation.from_quat(\
                                                utils.reverse_flip_quat(prev_state[3])
                                    ).as_euler('xyz', degrees=False)).tolist()+extra_dim[:])

                        action = controller.compute_control(prev_state)
                        sim.apply_control(action)
                        new_state, _ = sim.get_state()
                        
                        x['next_block_pos'].append(np.copy(new_state[0]).tolist()+extra_dim[:])
                        x['next_block_quat'].append(np.copy(Rotation.from_quat(\
                                                utils.reverse_flip_quat(new_state[1])
                                    ).as_euler('xyz', degrees=False)).tolist()+extra_dim[:])
                        
                        x['next_car_pos'].append(np.copy(new_state[2]).tolist()+extra_dim[:])
                        x['next_car_quat'].append(np.copy(Rotation.from_quat(\
                                                utils.reverse_flip_quat(new_state[3])
                                    ).as_euler('xyz', degrees=False)).tolist()+extra_dim[:])
                        
                        x['speed'].append(action[0])
                        x['steering_angle'].append(action[1])
                        
                        x_len_tracker+=1
                        
                        new_exp=False
                        
                    else:
                        state, _ = sim.get_state()
                        action = controller.compute_control(state)
                        sim.apply_control(action)
                    
                    sim._check_end_condition(time.time()-start > time_limit)
                    
                    if not sim.headless:
                        sim.render()
                
                fig=plt.figure()
                plt.xlim(-15,15)
                plt.ylim(-15,15)
                plt.plot([t[0] for t in x['prev_block_pos'][plt_count:]], [t[1] for t in x['prev_block_pos'][plt_count:]], 's', markersize=5,\
             markeredgewidth=1, markeredgecolor=None, alpha=0.2, label='Block Pos (x,y)')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.axes().set_aspect('equal')
                plt.legend()
                plt_count=len(x['prev_block_pos'])
                plt.savefig('datacol/json_files/R%d_T%d.png'%(sim.run_number, sim.trial_number))
                plt.close()
                
                if x_len_tracker>10000:
                    x = delete_common_instances(x)
                    save_json(x, controller.__class__.__name__, x_count, \
                              controller.speed, controller.angle , save_path)
                    x=Create_empty_x()
                    plt_count=0
                    x_count+=1
                    x_len_tracker=0
        
        m = sim.get_metrics()
        
        x['metrics'] = m
        
        sim.reset_sim()
        print('Complete!')
                
        x = delete_common_instances(x)
    
    except KeyboardInterrupt:
        print ("Keyboard interrupt called...")
    
    finally:
        if x!={}:
            save_json(x, controller.__class__.__name__, x_count, \
                      controller.speed, controller.angle, save_path)
            x_count+=1
            
