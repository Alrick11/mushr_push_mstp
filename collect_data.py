#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 01:12:23 2020

@author: alrick
"""
import multiprocessing as mp
from multiprocessing import Process
import argparse
import datacol
import src.sim as mps
from tqdm.auto import tqdm
import yaml

if __name__=="__main__":
    """
    Collect data for training. This script utilizes multiprocessing and 
    requires an input for number of parallel processes.
    """
    
    print ("Total CPUs: ", mp.cpu_count())
    
    parser=argparse.ArgumentParser(description="Argument parser for data collection")
    parser.add_argument("--cpu_count", type=int, default=1, \
                        help="Number of parallel processes you want running.")
    parser.add_argument ("--out_dir", type=str, default="datacol/json_files",\
                         help="Output dir of .json files")
    parser.add_argument("--runs", type=int, default=30, help="No. of runs")
    parser.add_argument("--trials", type=int, default=15, help="No of trials")
    parser.add_argument("--time_limit", type=int, default=10, \
                        help="Max allowable time to run one trial")
    parser.add_argument("--plot_steps", type=int, default=50,\
                        help="interval for plotting")
        
    args = parser.parse_args()
    
    #Raise an exception if cpu_count is greater than MP count
    if args.cpu_count > mp.cpu_count():
        raise Exception("Input cpu count greater than total exisiting cpus %d"%(mp.cpu_count()))
    
    #Load config file
    with open("config/config.yaml", 'r') as p:
        config=yaml.safe_load(p)
    config['num_runs']=args.runs
    config['num_trials']=args.trials
    config['headless']=True
    
    #Current controller
    controllers = [datacol.data.Random_policy(0,0), datacol.data.Random_policy(1,0),\
                    datacol.data.Random_policy(1,1), datacol.data.Random_policy(0,1)]
    
    #Define Controllers
    # controllers = [datacol.data.Random_policy(0,0), datacol.data.Random_policy(1,0),\
    #                 datacol.data.Random_policy(1,1), datacol.data.Random_policy(0,1),\
    #                 datacol.data.Curved_policy(0.5,0.05), datacol.data.Curved_policy(0.5,0.1),\
    #                 datacol.data.Curved_policy(0.5,-0.05), datacol.data.Curved_policy(0.5,-0.1),\
    #                 datacol.data.StraightPath_policy(v=0.4)]
    
    #Subset controllers    
    # controllers = [datacol.data.StraightPath_policy(v=0.6), datacol.data.StraightPath_policy(v=0.52),\
    #                datacol.data.StraightPath_policy(v=0.65)]
    #controllers = [datacol.data.Curved_policy(0.5, 0.002), datacol.data.Curved_policy(0.5, -0.002),\
    #                datacol.data.Curved_policy(0.5, 0.2), datacol.data.Curved_policy(0.5, 0.3)]
    # controllers = [datacol.data.Random_policy(1,1)]
    n_cont = len(controllers)
    
    simulators = [mps.MushrPushSim(config=config) for i in range(args.cpu_count)]
    
    #Initialize Variables
    i,prev_i,j=0,0,0
    indices=[i for i in range(args.cpu_count)]
    jobs=[]
    
    pbar = tqdm(total=n_cont, desc="Controllers Done", position=0, leave=True)
    
    while i<n_cont:
        pbar.update(i-prev_i)
        prev_i=i
        while j<args.cpu_count:
            p=Process(target=datacol.data.collect_data, args=(simulators[indices[0]],\
                                                              controllers[i],\
                                                    args.runs,args.trials, args.time_limit, \
                                                        args.out_dir))
            indices.pop(0)
            jobs.append(p)
            p.daemon=True
            p.start()
            i+=1
            if i>=n_cont: break
            j+=1
        
        #Join Processes
        for k in range(len(jobs)):
            jobs[k].join()

        while sum(j.is_alive() for j in jobs)==len(jobs):
            pass
        
        if len(indices)==0:
            remove_idxs=[]
            for k in range(len(jobs)):
                if not jobs[k].is_alive():
                    indices.append(k)
                    j-=1
                    
            jobs = [job for k,job in enumerate(jobs) if k not in indices]
            
    datacol.combine_data(args.out_dir)
    print ("\nDone Combining Data")
    datacol.visualize_data(args.out_dir, args.plot_steps)
    print ("\nDone making plots")
    print ("\nSplitting Data into train and test")
    datacol.split_data(args.out_dir)
    print ("\nDone Splitting Data")
    
                
