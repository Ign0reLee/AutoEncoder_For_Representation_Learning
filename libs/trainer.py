import numpy as np

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter, writer

from .layers import *
from .models import *
from .util import *

class AETrainer():
    def __init__(self, restart= False, in_channels=3, out_channels=3, nker=32, bnorm="bnorm", optimzer="adam",
     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), fn_loss=nn.MSELoss(), **kwargs):
        
        self.model = AutoEncoder(in_channels=in_channels, out_channels=out_channels, nker=nker, norm=bnorm)
        self.fn_loss = fn_loss.to("cuda:1")
        if optimzer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=kwargs["lr"], betas=(0.5, 0.999))
        
        self.st_epoch = 0
        self.device = device
        self.num_batch_train = kwargs["num_batch_train"]
        self.restart = restart
        self.ckpt_dir = kwargs["ckpt_dir"]

        self.writer_autoencoder = SummaryWriter(log_dir=kwargs["log_dir"])

        self.to_numpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 3, 1)

    def train(self, loader, num_epochs):

        if self.restart:
            self.model, self.optim, self.st_epoch = load(ckpt_dir= self.ckpt_dir, model=self.model, optim=self.optim)

        self.model.train()

        for epoch in range(self.st_epoch+1, num_epochs+1):

            train_loss = []

            for batch, data in enumerate(loader, 1):

                img = data['label'].to("cuda:0")
                output = self.model(img)
                self.optim.zero_grad()

                loss = self.fn_loss(output, img.to("cuda:1"))
                train_loss += [loss.item()]
                loss.backward()

                self.optim.step()

                print(f"TRAIN : {epoch} / {num_epochs} | BATCH : {batch} / {self.num_batch_train} | LOSS : {np.mean(train_loss):.4f}")

                if batch % 10 ==0:
                    output = self.to_numpy(output).squeeze()
                    output = np.clip(output, a_min=0, a_max=1)
                    id = self.num_batch_train * (epoch - 1)+ batch

                    self.writer_autoencoder.add_image("Input", img, id, dataformats="NCHW")    
                    self.writer_autoencoder.add_image("output", output, id, dataformats="NHWC")

            self.writer_autoencoder.add_scalar("Loss", np.mean(train_loss), epoch)
            save(ckpt_dir=self.ckpt_dir, model=self.model, optim=self.optim, epoch=epoch)
                
