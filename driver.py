import generator
import discriminator
import env_settings as config
import data_handler
from torch.cuda.amp import GradScaler
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
import torch 

class cGAN:
    def __init__(self):
        self.genM1=generator.generator().to(config.DEVICE)
        self.genM2=generator.generator().to(config.DEVICE)
        self.discM1=discriminator.Discriminator().to(config.DEVICE)
        self.discM2=discriminator.Discriminator().to(config.DEVICE)
        self.dataLoader=data_handler.loader()
        self.discScaler=GradScaler()
        self.genScaler=GradScaler()
        self.L1_loss=nn.L1Loss()
        self.mse=nn.MSELoss()
        self.opt_disc=optim.Adam(
                list(self.discM1.parameters()) + list(self.discM2.parameters()),
                lr=config.learning_rate,
                betas=(0.5,0.999)
                )
        self.opt_gen=optim.Adam(
                list(self.genM1.parameters()) + list(self.genM2.parameters()),
                lr=config.learning_rate,
                betas=(0.5,0.999)
                )

    def train(self):
        for epoch in range(config.epochs):
            for _,(data_point1,data_point2) in enumerate(self.dataLoader):

                data_point1=data_point1.to(config.DEVICE)
                data_point2=data_point2.to(config.DEVICE)
                
                # Discriminator loss wrt datapoint1. 
                fake_data_point1=self.genM1(data_point2)
                disc_data_point1_real=self.discM1(data_point1)
                disc_data_point1_fake=self.discM1(fake_data_point1.detach()) # after detach the tensor is removed from computation graph and gradient tracking for it is not done.
                disc_data_point1_real_loss=self.mse(disc_data_point1_real,torch.ones_like(disc_data_point1_real))
                disc_data_point1_fake_loss=self.mse(disc_data_point1_fake,torch.ones_like(disc_data_point1_fake))
                disc_data_point1_loss=(disc_data_point1_real_loss + disc_data_point1_fake_loss)/2


                # Discriminator loss wrt datapoint2.
                fake_data_point2=self.genM2(data_point1)
                disc_data_point2_real=self.discM2(data_point2)
                disc_data_point2_fake=self.discM2(fake_data_point2.detach()) # after detach the tensor is removed from computation graph and gradient tracking for it is not done.
                disc_data_point2_real_loss=self.mse(disc_data_point2_real,torch.ones_like(disc_data_point2_real))
                disc_data_point2_fake_loss=self.mse(disc_data_point2_fake,torch.ones_like(disc_data_point2_fake))
                disc_data_point2_loss=(disc_data_point2_real_loss + disc_data_point2_fake_loss)/2

                disc_loss=disc_data_point1_loss + disc_data_point2_loss


                self.opt_disc.zero_grad()
                self.discScaler.scale(disc_loss).backward()
                self.discScaler.step(self.opt_disc)
                self.discScaler.update()

                # adversial loss for generators
                disc_data_point1_fake=self.discM1(fake_data_point1)
                disc_data_point2_fake=self.discM2(fake_data_point2)
                gen_data_point1_loss=self.mse(disc_data_point1_fake,torch.ones_like(disc_data_point1_fake))
                gen_data_point2_loss=self.mse(disc_data_point2_fake,torch.ones_like(disc_data_point2_fake))

                # cycle loss
                data_point1_cycle=self.genM2(data_point1)
                data_point2_cycle=self.genM1(data_point2)
                data_point1_cycle_loss=self.L1_loss(data_point1,data_point1_cycle)
                data_point2_cycle_loss=self.L1_loss(data_point2,data_point2_cycle)

                # Identity loss is not calculated (for efficiency)

                # aggregate loss 
                gen_loss=gen_data_point1_loss + gen_data_point2_loss + data_point1_cycle_loss * 10 + data_point2_cycle_loss * 10

                self.opt_gen.zero_grad()
                self.genScaler.scale(gen_loss).backward()
                self.genScaler.step(self.opt_gen)
                self.genScaler.update()

            save_image(fake_data_point1 ,f"./data/generated/data_point1_{epoch}.png")
            save_image(fake_data_point2 ,f"./data/generated/data_point2_{epoch}.png")

            print(f"Epoch = {epoch}, generator loss = {gen_loss}, discriminator loss = {disc_loss}")
            self.save_checkpoint()
    def save_checkpoint(self):
        torch.save({
            "state_dict" : self.genM1.state_dict(),
            "optimizer" : self.opt_gen.state_dict()
        },"Generator_Model_1.tar.gz")
        torch.save({
            "state_dict" : self.genM2.state_dict(),
            "optimizer" : self.opt_gen.state_dict()
        },"Generator_Model_2.tar.gz")
        torch.save({
            "state_dict" : self.discM1.state_dict(),
            "optimizer" : self.opt_disc.state_dict()
        },"Discriminator_Model_1.tar.gz")
        torch.save({
            "state_dict" : self.discM2.state_dict(),
            "optimizer" : self.opt_disc.state_dict()
        },"Discriminator_Model_2.tar.gz")

if __name__=='__main__':
    cycleGAN=cGAN()
    cycleGAN.train()





