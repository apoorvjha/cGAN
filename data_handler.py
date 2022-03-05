from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Resize, Compose, ToTensor
import env_settings as config
from numpy import array, float32

class DataHandler:
    def __init__(self,data_point1_dir='./data/data_point1/trainA/',generated_dir='./data/generated/',data_point2_dir='./data/data_point2/trainB/'):
        self.data_point1_dir=data_point1_dir
        self.generated_dir=generated_dir
        self.data_point2_dir=data_point2_dir
    def data_loader(self):
        self.data_point1=[]
        self.data_point2=[]
        for i in os.listdir(self.data_point1_dir):
            self.data_point1.append(array(Image.open(self.data_point1_dir + i).convert("RGB"),dtype=float32))
        for i in os.listdir(self.data_point2_dir):
            self.data_point2.append(array(Image.open(self.data_point2_dir + i).convert("RGB"),dtype=float32))
        self.data_point1=array(self.data_point1)
        self.data_point2=array(self.data_point2)
        self.data_point1=self.preprocess(torch.Tensor(self.data_point1.reshape(-1,self.data_point1.shape[3],self.data_point1.shape[1],self.data_point1.shape[2])))
        self.data_point2=self.preprocess(torch.Tensor(self.data_point2.reshape(-1,self.data_point2.shape[3],self.data_point2.shape[1],self.data_point2.shape[2])))
    def preprocess(self,image,dimension=(64,64)):
        transforms=Compose([
            Resize(size=dimension),
            Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),inplace=True)
        ])
        image=transforms(image)
        return image
    def getLength(self):
        return max(len(os.listdir(self.data_point1_dir)),len(os.listdir(self.data_point2_dir)))

    def getData(self,idx):
        return self.data_point1[idx % len(os.listdir(self.data_point1_dir))], self.data_point2[idx % len(os.listdir(self.data_point2_dir))] 


class DatasetTensor(Dataset):
    def __init__(self):
        self.data_handler=DataHandler()
        self.data_handler.data_loader()
    def __len__(self):
        return self.data_handler.getLength()
    def __getitem__(self,idx):
        return self.data_handler.getData(idx)

def loader():
    dataset=DatasetTensor()
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)



        
        

