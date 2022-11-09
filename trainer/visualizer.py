import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from .parameters import VizParameters


class Visulaize3D:
    def __init__(self, params: VizParameters, model_name:str, train_samples:int) -> None:
        self.params= params
        self.model_name, self.train_samples= model_name, train_samples

        self.set_style()
        if not os.path.exists(self.params.dir):
            os.makedirs(self.params.dir)
        
    def plot(self, idx, y_PRED, y, pos):
        # is called multiple times for plotting 
        y_NEW= y.clone()
        # add displacement to get the coords
        y_NEW[..., 0:3]+= pos
        y_PRED[..., 0:3]+= pos 
        delta= y_NEW-y_PRED
        
        fig=plt.figure(figsize=(15, 7))
        self.manage_subplot(y_NEW, y_PRED, delta, 1, fig)
        self.adjust_layout(fig)

        if self.params.save:
            path= rf"m-{self.model_name}_ts-{self.train_samples}_s-{idx}.png"
            path= os.path.join(self.params.dir, path)
            self.save(path)
        
        if self.params.show:
            plt.show()
        
        self.reset()


    def manage_subplot( self, y_NEW, y_PRED, delta, i, 
                        fig, shape=(1,3)):
        
        _i= (i-1)*3
        ax= fig.add_subplot(*shape, _i+1, projection='3d')
        self.visualize3D(y_NEW, name="Target", ax= ax, fig=fig)
        
        ax= fig.add_subplot(*shape, _i+2, projection='3d')
        self.visualize3D(y_PRED, name="Prediction", ax= ax, fig=fig)
        
        ax= fig.add_subplot(*shape, _i+3, projection='3d')
        self.visualize3D(delta, name="Difference", ax= ax, fig=fig)

        return fig

    def adjust_layout(self, fig):
        fig.tight_layout()
        plt.subplots_adjust(left=0.241,\
                    right= 0.954, \
                    bottom=0.021,\
                    top=0.828,\
                    wspace=0.12,\
                    hspace=0.071)
        # plt.subplot_tool()

    @staticmethod
    def visualize3D(data, name, ax= None, fig= None):
        if isinstance(data, torch.Tensor):
            data= data.cpu().detach().numpy()

        if ax is None:
            fig = plt.figure()
            ax= fig.add_subplot(projection='3d')

        p= ax.scatter(data[..., 0], data[..., 1], data[..., 2], c=data[..., 3], cmap= 'magma_r', s= 10, alpha=0.7) # facecolor= 'C1', edgecolors='none', viridis_r
        cbar= fig.colorbar(p, label= "Thickness", aspect= 40, shrink=0.5)

        ax.set(
                xlim=[data[...,0].min(), data[...,0].max()],
                ylim=[data[...,1].min(), data[...,1].max()],
                xlabel= 'x', ylabel= 'y', zlabel= 'z',
                title= name)
        ax.view_init(elev= 16, azim= -116)
        return ax 

    @staticmethod
    def set_style():
        mpl.rcParams['figure.autolayout']= True
        plt.style.use('ggplot') # seaborn-bright ggplot

    def close(self):
        if self.params.show:
            plt.show()

    def save(self, path):
        plt.savefig(path, dpi=200)
    
    def reset(self):
        plt.clf()
        plt.cla()
        plt.close()

    

