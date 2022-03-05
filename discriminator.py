import torch
import torch.nn as nn
from torch.cuda.amp import autocast

# Autocast is used to cast the operations to their compatible datatypes automatically.

class Discriminator(nn.Module):
    def __init__(self,img_channels=3,n_hidden_layers=[64,128,256,512]):
        super().__init__()
        self.Input=nn.Sequential(
                nn.Conv2d(in_channels=img_channels,out_channels=n_hidden_layers[0],
                kernel_size=(4,4),stride=2,padding_mode='reflect',padding=1,bias=True),
                nn.LeakyReLU(0.2)
                )
        hidden_layers=[]
        in_channels=n_hidden_layers[0]
        for n in range(len(n_hidden_layers[1:])):
            if n!=len(n_hidden_layers[1:]):
                hidden_layers.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels=in_channels,out_channels=n_hidden_layers[n],
                                stride=(2,2),padding_mode='reflect',padding=1,kernel_size=(4,4),bias=True),
                            nn.InstanceNorm2d(n_hidden_layers[n]),
                            nn.LeakyReLU(0.2)
                            )
                        )
            else:
                hidden_layers.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels=in_channels,out_channels=n_hidden_layers[n],
                                stride=(1,1),padding_mode='reflect',padding=1,kernel_size=(4,4),bias=True),
                            nn.InstanceNorm2d(n_hidden_layers[n]),
                            nn.LeakyReLU(0.2)
                            )
                        )
            in_channels=n_hidden_layers[n]
        hidden_layers.append(
                nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=(4,4),stride=1,padding=1,
                    padding_mode='reflect',bias=True)
                )
        self.model=nn.Sequential(*hidden_layers)  # unfold the hidden_layers
    @autocast()
    def forward(self,x):
        x=self.Input(x)
        x=self.model(x)
        return nn.Sigmoid()(x)

def test():
    disc_obj=Discriminator()
    x=torch.randn(2,3,256,256)
    pred=disc_obj(x)
    print(pred.size())

if __name__=='__main__':
    test()




