"ToDo give a docstring"
import os
from datetime import datetime
from random import sample

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from .common import sampler
from .parameters import Parameters
from .visualizer import Visulaize3D


def now():
    now_ = datetime.now()
    return now_.strftime("%m_%d_%H_%M")


class CommonBase:

    def __init__(self, params: Parameters, model, callbacks={}) -> None:
        self.model= model
        self.model.to(params.hyper.device)
        self.params = params
        self.params.hyper.model_name= type(self.model).__name__

        self.time_ext= now()
        self.base_dir= rf"{self.params.data.ckpt_base_dir}/saved_data"
        self.log_path = rf"{self.base_dir}/{self.time_ext}/logs"
        self.checkpoint_path = rf"{self.base_dir}/{self.time_ext}/checkpoints"
        self.params.viz.dir = rf"{self.base_dir}/{self.time_ext}/visulaizations"
        self.callbacks = callbacks



    def dataloader(self, dataset):
        dataloader_ = DataLoader(dataset, batch_size=self.params.hyper.batch_size,
                                 shuffle=True, pin_memory=True,
                                 num_workers=4, persistent_workers=True)
        return dataloader_

    @classmethod
    def load_from_checkpoint(cls, checkpoint, params, model, callbacks={}, **kwargs):
        
        ################# loading checkpoint ##################

        def optim_loader(self):
            self.start_epoch = checkpoint['epoch']+1
            if params.hyper.load_opt_state:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  
        
        callbacks['optim_loader'] = optim_loader

        self = cls(params=params, model=model, callbacks=callbacks, **kwargs)
        
        self.time_ext = checkpoint['time_ext']
        self.log_path = rf"{self.base_dir}/{self.time_ext}/logs"
        self.checkpoint_path = rf"{self.base_dir}/{self.time_ext}/checkpoints"
        self.params.viz.dir = rf"{self.base_dir}/{self.time_ext}/visulaizations"
        return self


    def save_checkpoint(self, epoch, model_name, train_size, loss):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        path = rf"m-{model_name}_e-{epoch}_ts-{train_size}_l-{loss:.2f}.pt"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'params_state_dict': self.params.state_dict(),
                    'time_ext': self.time_ext,
                    },  os.path.join(self.checkpoint_path, path))
        self.params.save_human_readable(self.checkpoint_path)

class TrainerBase(CommonBase):

    def __init__(self, params: Parameters, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.start_epoch = 1

        ################## set configurations #################
        self.optimizer_config()
        self.loss_config()

    def train(self, trainset, valset):

        self.train_loader = self.dataloader(trainset)
        self.val_loader = self.dataloader(valset)

        optim_loader= self.callbacks.get('optim_loader')
        if optim_loader is not None:
            optim_loader(self)
        
        self.model.train()
        
        del trainset, valset    # to save memory

        writer = SummaryWriter(self.log_path, flush_secs=300)

        train_size = self.params.data.n_train
        model_name = self.params.hyper.model_name
        end_epoch = self.params.hyper.num_epochs+1

        epoch_loss = None
        with tqdm(range(self.start_epoch, end_epoch), unit='epoch', desc="Training", colour='red', dynamic_ncols=True) as tepochs:
            for epoch in tepochs:
                l_t = []
                l_v = []
                # with tqdm(self.trainloader, unit="batch") as tepoch:
                for i, data in enumerate(self.train_loader):
                    j = epoch*self.params.hyper.batch_size + i
                    # with torch.autograd.detect_anomaly():
                    loss = self.train_step(data)
                    l_t.append(loss)
                    writer.add_scalar('train loss', loss, j)
                    # tepochs.set_description(f"Epoch {epoch}")

                if epoch%2 == 0:
                    with torch.no_grad():
                        for data in self.val_loader:
                            loss_v = self.val_step(data)
                            l_v.append(loss_v)
                            writer.add_scalar('validation loss', loss_v, j)

                writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
                
                epoch_loss = torch.tensor(l_t).mean().item()
                epoch_loss_v = torch.tensor(l_v).mean().item()
                writer.add_scalars('Train/Val loss', {"train": epoch_loss, 
                                                      "val": epoch_loss_v}, epoch)

                tepochs.set_postfix(loss=epoch_loss, refresh=False)
                
                if epoch%20 == 0 or epoch==end_epoch-1:
                    self.save_checkpoint(epoch, model_name, train_size, loss)

                if self.params.hyper.use_lrscheduling:
                    self.scheduler.step(epoch_loss)
        writer.close()
        print("Training Done b^.^d")
        return epoch_loss

    def optimizer_config(self):
        optimizer_list = {'Adam': optim.Adam, 'SGD': optim.SGD}
        self.optimizer = optimizer_list[self.params.hyper.optimizer](self.model.parameters(), lr=self.params.hyper.lr)
        
        step_size = int(self.params.hyper.num_epochs/100)
        # self.scheduler= optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma= 0.9)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience= 2*step_size, factor=0.6, verbose=True, min_lr=1e-8, cooldown=1)
    
    def loss_config(self):
        self.criterion = nn.MSELoss()


    @torch.no_grad()
    def test(self, testset, output_norm=None):
        self.test_loss_config()
        self.output_norm= output_norm
        # setting batchsize to 1 
        train_batch_size = self.params.hyper.batch_size
        self.params.hyper.batch_size = 1

        visualize = self.params.viz.visualize
        
        self.testloader = self.dataloader(testset)
        self.model.to(self.params.hyper.device).eval()

        if visualize:
            train_size = self.params.data.train_size
            model_name = self.params.hyper.model_name
            self.viz = Visulaize3D(self.params.viz, model_name, train_size)

            # sample l random indices for visualization later
            if self.params.viz.indices is None:
                indices= sample(range(0, len(self.testloader)), self.params.viz.l)
            else: 
                indices= self.params.viz.indices 
            sample_next = sampler(indices)
            j = sample_next()  # calling next
        
        writer = SummaryWriter(self.log_path, flush_secs=300)

        with tqdm(self.testloader, unit="batch", desc="Testing", colour="red") as tloader:
            for i, data in enumerate(tloader):
                loss, y_PRED = self.test_step(data)
                tloader.set_postfix(loss=loss, refresh=False)
                writer.add_scalars('test loss', loss, i)
                if visualize and i == j:
                    self.viz_step(i, y_PRED, data)
                    j = sample_next()
                    # if samples are exhausted turnoff visulaisation
                    if j==-1:
                        visualize = False

        self.params.hyper.batch_size = train_batch_size
        writer.close()
        if visualize:
            self.viz.close()
        print("Testing Done :D")

        return loss
    
    def test_loss_config(self):
        self.loss_config()

    def train_step(self, data):
        raise NotImplementedError

    def val_step(self, data):
        raise NotImplementedError

    def test_step(self, data):
        raise NotImplementedError
    
    def viz_step(self, idx, y_PRED, data):
        raise NotImplementedError
