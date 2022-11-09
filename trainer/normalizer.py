import numpy as np
import pickle
import torch
import os.path as osp
import torch 

class Normalizer: 
    def __init__(self, name:str):
        self.name= name
        self.min_, self.max_= None, None
        self.range_= None
        self.samples_seen= 0
    
    def __call__(self, x):
        return self.transform(x)

    def fit(self, x):
        if isinstance(x, np.ndarray):
            return self.fit_numpy(x)
        else:
            return self.fit_torch(x)

    def fit_numpy(self, x:np.ndarray):
        min, max= x.min(axis=0, keepdims= True), x.max(axis=0, keepdims= True)
        if self.samples_seen == 0: 
            #First pass 
            self.min_, self.max_= min, max

        else: 
            self.min_, self.max_= np.minimum(min, self.min_), np.maximum(max, self.max_)
        
        self.range_= self.max_-self.min_ 
        self.samples_seen+=1 
    
    def fit_torch(self, x:torch.tensor):
        min, max= x.min(dim=0, keepdims= True).values, x.max(dim=0, keepdims= True).values
        if self.samples_seen == 0: 
            #First pass 
            self.min_, self.max_= min.numpy(), max.numpy()

        else: 
            self.min_, self.max_= np.minimum(min, self.min_), np.maximum(max, self.max_)
        
        self.range_= self.max_-self.min_ 
        self.samples_seen+=1

    def transform(self, x):
        min, max= torch.from_numpy(self.min_), torch.from_numpy(self.max_)
        range= torch.from_numpy(self.range_)

        x= (x-min)/(range)
        return x
        
    def inverse(self, y):
        device= y.device
        y= y * torch.from_numpy(self.range_).to(device) + torch.from_numpy(self.min_).to(device)
        return y

    def save_state(self, dir:str= ""):
        dir= osp.join(dir,self.name+"_norm.pkl")
        with open(dir, "wb") as pickle_file:
            pickle.dump(self, pickle_file)
    
    @classmethod
    def load_state(cls, dir:str= ""): #todo give an appropriate def val
        with open(dir, "rb") as pickle_file:
            return pickle.load(pickle_file)
