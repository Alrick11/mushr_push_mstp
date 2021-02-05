#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:36:34 2020

@author: alrick
"""
#Main file to run
import argparse, os

from TrainEval import train
from DataLoader import BlockPosDataLoader
from Postanalysis import Generate_metrics

if __name__=="__main__":
    """
    Main function. Run this function for training and testing.
    """
    
    parser=argparse.ArgumentParser(description="Hyperparams and Params for training")
    parser.add_argument('--data_dir',type=str,default='/home/ugrads/hard_data/Alrick/train_data',
                        help='Main directory where train.csv.gz, test.csv.gz files are stored')
    parser.add_argument('--home_dir',type=str,default='/home/ugrads/WRK/Alrick/mushr_push_sim/train',
                        help='Home directory of the main.py file')
    parser.add_argument('--traj_save_addr',type=str,default='/home/ugrads/hard_data/Alrick/TrajVis',
                        help='Directory to save Trajectories after generating the model')
    parser.add_argument('--model_path',type=str,default='/home/ugrads/hard_data/Alrick/PretrainedModel/',
                        help='Directory where pretrained model is to be saved')
    parser.add_argument('--train',type=bool,default=False,help='Whether to train or simply perform Model analysis')
    #The above parameters need to change ...
    
    parser.add_argument('--train_file',type=str,default='')
    parser.add_argument('--test_seen_file',type=str,default='')
    parser.add_argument('--test_unseen_file',type=str,default='')
    parser.add_argument('--writer_loc',type=str,default='')
    parser.add_argument('--is_lstm',type=bool,default=False)
    
    #Hyper-parameters
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--weight_decay',type=float,default=0.0000)
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--interval',type=int,default=60000000000)
    parser.add_argument('--write_interval',type=int,default=5000,help="Epochs for writing")
    
    #Model parameters
    parser.add_argument('--x_enc_feat',type=int,default=100)
    parser.add_argument('--u_enc_feat',type=int,default=100)
    parser.add_argument('--dropout',type=float,default=0.2)
    parser.add_argument('--batch_size',type=int,default=500)
    parser.add_argument('--sequence_len',type=int,default=1,
                        help='LSTM sequence length')
    parser.add_argument('--n_layers',type=int,default=3,
                        help='Number of LSTM layers')
    parser.add_argument('--relative',type=bool,default=True,
                        help='Load block pose relative to car pose')
    
    args=parser.parse_args()
    
    args.train_file = os.path.join(args.data_dir,'train_f.csv.gz')
    args.test_seen_file=os.path.join(args.data_dir, 'test_seen_f.csv.gz')
    args.test_unseen_file=os.path.join(args.data_dir, 'test_unseen_f.csv.gz')
    args.writer_loc=os.path.join(args.home_dir,'TensorboardVisuals')
    
    #Clear Trajectory addr
    os.system("rm -rf %s/*"%(args.traj_save_addr))
    
    if not os.path.exists(args.train_file) and not os.path.exists(args.test_seen_file) and not os.path.exists(args.test_unseen_file):
        print ("Path does not exist")
        exit(0)
    
    if args.train:
        train_loader = BlockPosDataLoader(args.train_file, args.sequence_len, args.batch_size, rel=args.relative)
        test_seen_loader = BlockPosDataLoader(args.test_seen_file, args.sequence_len, args.batch_size, rel=args.relative)
        test_unseen_loader = BlockPosDataLoader(args.test_unseen_file, args.sequence_len, args.batch_size, rel=args.relative)
            
        train(train_loader, test_seen_loader, test_unseen_loader, **vars(args))
        
    else:
        Generate_metrics(is_traj=False, **vars(args))
        