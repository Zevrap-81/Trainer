import torch
from torch import nn
from torch.utils.data import TensorDataset, Subset

from class_resolver.contrib.torch import activation_resolver
import trainer
from trainer.normalizer import Normalizer
from trainer.parameters import Parameters
from trainer.trainer import TrainerBase

# torch.autograd.set_detect_anomaly(True)

class Trainer(TrainerBase):

    def __init__(self, params: Parameters, model: nn.Module, **kwargs) -> None:
        super().__init__(params, model=model, **kwargs)

    def train_step(self, data):
        x, y= data
        x = x.to(self.params.hyper.device)
        y = y.to(self.params.hyper.device)
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, data):
        x, y= data
        x = x.to(self.params.hyper.device)
        y = y.to(self.params.hyper.device)
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return loss.item()
    
    def test_step(self, data):
        x, y= data
        x = x.to(self.params.hyper.device)
        y = y.to(self.params.hyper.device)
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return loss.item(), pred #should return both loss and the prediction for visualisation
    
if __name__== "__main__":
    #Parameters Setup
    params= Parameters()
    params.data.n_train= 200
    params.data.split_ratio=0.8
    params.hyper.device= "cuda:0"
    params.viz.visualize= False
    
    #data prep
    params.hyper.in_channel=3
    params.hyper.out_channel=1 

    x= torch.randn(params.data.n_train, params.hyper.in_channel)
    y= torch.randn(params.data.n_train, params.hyper.out_channel)
    input_norm= Normalizer("input")
    output_norm=Normalizer("output")
    input_norm.fit(x); output_norm.fit(y)
    x= input_norm(x); y= output_norm(y)

    dataset= TensorDataset(x, y)
    l= int(params.data.n_train*params.data.split_ratio)
    trainset, valset= Subset(dataset, list(range(0,l))), Subset(dataset, list(range(l,params.data.n_train)))

    #Instantiate the model
    model = nn.Sequential(nn.Linear(params.hyper.in_channel, params.hyper.hidden_channel),
                          activation_resolver.make(params.hyper.act),
                          nn.Linear(params.hyper.hidden_channel, params.hyper.out_channel))
    
    
    #Instantiate the trainer
    trainer= Trainer(params, model)
    # training and validation
    trainer.train(trainset, valset)
    #testing and postprocessing
    trainer.test(valset, output_norm)
