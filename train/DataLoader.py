#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:35:28 2020

@author: alrick
"""
#Dataloader
import torch
from torch.utils.data import Dataset
from pandas import read_csv
from ast import literal_eval

class BlockPosDataLoader(Dataset):
    def __init__(self, filename, sequence_len, batch_size, rel=False):
        """
        Loading the Dataloader for training.
        """
        super(BlockPosDataLoader, self).__init__()
        
        self.seq_len=sequence_len
        self.batch_size=batch_size
        
        print ('Reading csv file')
        reader=read_csv(filename)
        print ('Done reading csv file')
        print ("Prev block pos: ", reader['prev_block_pos'][0], type(reader['prev_block_pos'][0]))
        print ("Prev block quat: ", reader['prev_block_quat'][0], type(reader['prev_block_quat'][0]))
        print ("Next block pos: ", reader['next_block_pos'][0], type(reader['next_block_pos'][0]))
        
        n=len(reader['prev_block_pos'])
        
        print ("Total n (before reduction): ", n)
        n-=n%(batch_size*sequence_len)
        self.size=n
        
        val=n//(batch_size)
        chunks=int(n/(batch_size*sequence_len))
        self.chunks=chunks
        
        start=0
        
        print ('Final len of dataset: ', n)
        print ('Val (Final_len//batch_size): ', val)
        
        self.data_b, self.data_c, self.data_u, self.labels_p=[[] for i in range(chunks)],[[] for i in range(chunks)],\
                                    [[] for i in range(chunks)], [[] for i in range(chunks)]
        
        print ('Chunks (Val // Sequence_len): ', chunks)
        
        #Divide into batches
        idx=0
        print ("Loading DataLoader ...")
        while n-idx*val > 0:            
            temp_bpp,temp_bnp,temp_cpp,temp_u=[],[],[],[]
            
            for j in range(start, start+val):
                if rel:
                    b_p = literal_eval(reader['prev_block_pos'][j])
                    b_q = literal_eval(reader['prev_block_quat'][j])
                    c_p = literal_eval(reader['prev_car_pos'][j])
                    c_q = literal_eval(reader['prev_car_quat'][j])
                    
                    b_n_p = literal_eval(reader['next_block_pos'][j])
                    b_n_q = literal_eval(reader['next_block_quat'][j])
                    c_n_p = literal_eval(reader['next_car_pos'][j])
                    c_n_q = literal_eval(reader['next_car_quat'][j])
                    
                    b_p = [x-y for x,y in zip(b_p[:2], c_p[:2])]+\
                                        [b_q[2]-c_q[2]]+\
                                            [b_q[-1]]
                                            
                    b_l = [x-y for x,y in zip(b_n_p[:2], c_n_p[:2])]+\
                                        [b_n_q[2]-c_n_q[2]]
                    
                    temp_bpp.append(b_p)
                    
                    temp_bnp.append(b_l)
                    
                    temp_cpp.append(c_p[:2]+c_q[2:])
                else:
                    temp_bpp.append(literal_eval(reader['prev_block_pos'][j])[:2] + \
                                    literal_eval(reader['prev_block_quat'][j])[2:])
                    
                    temp_bnp.append(literal_eval(reader['next_block_pos'][j])[:2]+\
                                    [literal_eval(reader['next_block_quat'][j])[2]])
                    
                    temp_cpp.append(literal_eval(reader['prev_car_pos'][j])[:2]+\
                                    literal_eval(reader['prev_car_quat'][j])[2:])
                
                temp_u.append((float(reader['speed'][j]), float(reader['steering_angle'][j])))
            
            s_idx=0
            
            
            for i in range(chunks):
                self.data_b[i].append(temp_bpp[s_idx:s_idx+sequence_len])
                
                self.data_c[i].append(temp_cpp[s_idx:s_idx+sequence_len])
                    
                self.data_u[i].append(temp_u[s_idx:s_idx+sequence_len])
                    
                self.labels_p[i].append(temp_bnp[s_idx:s_idx+sequence_len])
                    
                s_idx+=sequence_len
            
            reader=reader.drop(list(range(start, start+val)))
            start+=val
            idx+=1
        
        self.labels_p=torch.tensor(self.labels_p, dtype=torch.float)
        self.data_b=torch.tensor(self.data_b, dtype=torch.float)
        self.data_c=torch.tensor(self.data_c, dtype=torch.float)
        self.data_u=torch.tensor(self.data_u, dtype=torch.float)
        
        print ("Data, Label shapes: ", self.data_b.shape, self.data_c.shape, self.data_u.shape, self.labels_p.shape)
        
    def __len__(self):
        return self.chunks
    
    def __getitem__(self, idx):
        return self.data_b[idx,:], self.data_c[idx,:], self.data_u[idx,:], self.labels_p[idx,:]
    
if __name__=="__main__":
    """
    Testing Dataloader
    """
    
    filepath='/home/ugrads/hard_data/Alrick/train_data/train.csv.gz'
    dataloader = BlockPosDataLoader(filepath, 5, 10)
    
    print (len(dataloader))
    
    for i, (data_b, data_c, data_u, labels_p) in enumerate(dataloader):
        print ("\nData_b:", data_b[0])
        
        print (i, data_b.shape, data_c.shape, data_u.shape, labels_p.shape)
    
    print ("Value of i: %d\n"%(i))
    print ("data_b\n", data_b[0], '\n\n')
    print ("data_c\n", data_b[1], '\n\n')
    print ("data_u\n", data_b[2], '\n\n')
    print ("labels_p\n", data_b[3], '\n\n')