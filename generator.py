import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class convLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,sample_mode):
        assert sample_mode=='up' or sample_mode=='down', "Unrecognized value for sample_mode. Only 'up' and 'down' values are acceptable!"
        super().__init__()
        if sample_mode=='down':
            self.layer=nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=1,
                        stride=stride,
                        padding_mode='reflect',
                        ),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                    )
        else:
            self.layer=nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=1,
                        stride=stride,
                        output_padding=1,
                        ),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                    )
    @autocast()
    def forward(self,x):
        return self.layer(x)
        

class generator(nn.Module):
    def __init__(self,img_channels=3,n_hidden_layers=[64,128,256],n_residual_layers=9):
        super().__init__()
        self.Input=nn.ModuleList(
                [
                convLayer(
                    in_channels=img_channels,
                    out_channels=n_hidden_layers[0],
                    kernel_size=(7,7),
                    sample_mode='down',
                    stride=1
                    ),
                convLayer(
                    in_channels=n_hidden_layers[0],
                    out_channels=n_hidden_layers[1],
                    kernel_size=(3,3),
                    sample_mode='down',
                    stride=2
                    ),
                convLayer(
                    in_channels=n_hidden_layers[1],
                    out_channels=n_hidden_layers[2],
                    kernel_size=(3,3),
                    sample_mode='down',
                    stride=2
                    )
                ]
                )
        self.residualLayer=nn.ModuleList(
                [
                    convLayer(
                    in_channels=n_hidden_layers[2],
                    out_channels=n_hidden_layers[2],
                    kernel_size=(3,3),
                    sample_mode='down',
                    stride=1
                    ) for _ in range(n_residual_layers)
                ]
                )
        self.Output=nn.ModuleList(
                [
                    convLayer(
                        in_channels=n_hidden_layers[2],
                        out_channels=n_hidden_layers[1],
                        sample_mode='up',
                        kernel_size=(3,3),
                        stride=2
                        ),
                    convLayer(
                        in_channels=n_hidden_layers[1],
                        out_channels=n_hidden_layers[0],
                        sample_mode='up',
                        kernel_size=(3,3),
                        stride=2
                        ),
                    convLayer(
                        in_channels=n_hidden_layers[0],
                        out_channels=img_channels,
                        sample_mode='down',
                        kernel_size=(7,7),
                        stride=1
                        )
                    ])
    @autocast()
    def forward(self,x):
        for layer in self.Input:
            x=layer(x)
        for layer in self.residualLayer:
            x=x+layer(x)
        for layer in self.Output:
            x=layer(x)
        image=torch.tanh(x)
        return torch.nn.functional.pad(image,(4,4,4,4),'reflect')

def test():
    genM=generator()
    x=torch.randn(2,3,256,256)
    pred=genM(x)
    print(pred.size())
    
if __name__=='__main__':
    test()
