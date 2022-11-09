import os
from dataclasses import dataclass
import torch 
import torch.nn as nn

def get_idle_device():
    """Get the most idle device on the network"""
    device_= f"cuda:{0}"
    util= torch.cuda.utilization(device_)
    for i in range(1, torch.cuda.device_count()):
        if torch.cuda.utilization(i) < util:
            device_= f'cuda:{i}'
    
    return device_

@dataclass
class BaseParameters:
    @classmethod
    def load_from_checkpoint(cls, dict):
        return cls(**dict)

    def state_dict(self):
        return self.__dict__


@dataclass
class DataParameters(BaseParameters):
    data_dir: str= r"data"
    split_ratio: float= 0.8
    n_train: int= 1000

    @property
    def train_size(self):
        return int(self.n_train*self.split_ratio)

@dataclass
class Hyperparameters(BaseParameters):

    model_name: str = None
    in_channel: int= 3
    hidden_channel: int= 128
    out_channel: int= 4
    act: nn.ReLU= nn.ReLU()

    optimizer: str= 'Adam'
    lr: float= 1e-2

    use_lrscheduling: bool= True
    load_opt_state: bool = True
    
    device_: str= None 
    batch_size: int= 75
    num_epochs: int= 5
    ####################################################################


    @property
    def device(self):
        """Method to automatically select the best device"""

        if self.device_ is not None:
            return self.device_
        
        #if gpu not available
        if not torch.cuda.is_available():
            self.device_= 'cpu'
            print(f"Device is set as {self.device_}")
            return self.device_

        #select the idle gpu if more than one available
        else:
            self.device_= get_idle_device()
            print(f"Device is set as {self.device_}")
            return self.device_

    @device.setter
    def device(self, val:str):
        self.device_= val 


@dataclass
class VizParameters(BaseParameters):
    visualize: bool= True
    indices: list= None
    l: int= 30
    dir: str= "visualisations"
    save: bool= True
    show: bool= False 


class Parameters:
    def __init__(self,  data=DataParameters(), 
                        hyper=Hyperparameters(), 
                        viz=VizParameters()) -> None:
        self.data= data
        self.hyper= hyper
        self.viz= viz

    @classmethod
    def load_from_checkpoint(cls, ckpt):
        # https://www.pythontutorial.net/python-oop/python-__new__/#:~:text=Summary-,The%20__new__()%20is%20a%20static%20method%20of%20the%20object,to%20initialize%20the%20object's%20attributes.
        params=  object.__new__(cls)
        params.data= DataParameters.load_from_checkpoint(ckpt["data"])
        params.hyper= Hyperparameters.load_from_checkpoint(ckpt["hyper"])
        params.viz= VizParameters.load_from_checkpoint(ckpt["viz"])
        return params

    def state_dict(self):
        return {"data": self.data.state_dict(), 
                "hyper": self.hyper.state_dict(), 
                "viz": self.viz.state_dict()}

    def save_human_readable(self, path):
        file = path + "/" + "parameters.txt"
        
        lines = [rf"learning rate : {self.hyper.lr}", 
                 rf"model_name : {self.hyper.model_name}", 
                 rf"batch_size : {self.hyper.batch_size}", 
                 rf"train/val split: {self.data.split_ratio}"]
        with open(file, 'w') as f:
            for line in lines:
                f.writelines(line)
        

if __name__=="__main__":
    params= Parameters()
    
    # testing loading and saving functionality
    params.data.split_ratio= 0.1

    torch.save({'params': params.state_dict()}, 'params.pt')
    del params

    ckpt= torch.load('params.pt')

    par= Parameters.load_from_checkpoint(ckpt['params'])
    print(par.data.split_ratio) 

    #cleanup
    os.remove('params.pt')