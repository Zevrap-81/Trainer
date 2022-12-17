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
    
    def __call__(self, x, inverse= False, batched=True):
        if batched:
            shape= x.shape
            x= x.reshape(-1, shape[-1])

        if not inverse:
            x= self.transform(x)
        else:
            x= self.inverse(x)
        
        if batched:
            x= x.reshape(shape)

        return x

    def fit(self, x, batched=True):
        if batched:
            batch_size=x.shape[0]
            shape= x.shape
            x= x.reshape(-1, shape[-1])

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
        self.__handling_zeros()

        min, max= torch.from_numpy(self.min_), torch.from_numpy(self.max_)
        range= torch.from_numpy(self.range_)

        x= (x-min)/(range)
        return x
        
    def inverse(self, y):
        device= y.device
        y= y * torch.from_numpy(self.range_).to(device) + torch.from_numpy(self.min_).to(device)
        return y

    def __handling_zeros(self):
        #handling zeroes in range
        constant_mask = self.range < 10 * np.finfo(self.range.dtype).eps
        self.range[constant_mask] = 1.0
        
    def save_state(self, dir:str= ""):
        dir= osp.join(dir,self.name+"_norm.pkl")
        with open(dir, "wb") as pickle_file:
            pickle.dump(self, pickle_file)
    
    @classmethod
    def load_state(cls, dir:str= ""): #todo give an appropriate def val
        with open(dir, "rb") as pickle_file:
            return pickle.load(pickle_file)
