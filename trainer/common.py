from random import sample
import torch

def load_checkpoint(load_path):
    ckpt = torch.load(load_path)
    model_ckpt = ckpt['model_state_dict']
    params_ckpt = ckpt['params_state_dict']
    ckpt.pop('params_state_dict')
    ckpt.pop('model_state_dict')
    return params_ckpt, model_ckpt, ckpt 

def sampler(iterable: list):
    iterable.sort()
    idx= iter(iterable)  

    def next_():
        return next(idx, -1)
    return next_ 
