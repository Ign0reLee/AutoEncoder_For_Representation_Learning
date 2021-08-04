import numpy as np

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter, writer

from .layers import *
from .models import *
from .util import *

class AETrainer():
    def __init__(self,test=False, restart= False, in_channels=3, out_channels=3, nker=32, bnorm="bnorm", optimzer="adam",
     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), fn_loss=nn.MSELoss(), **kwargs):

        self.ckpt_dir = kwargs["ckpt_dir"]
        self.to_numpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 3, 1)
        self.model = AutoEncoder(in_channels=in_channels, out_channels=out_channels, nker=nker, norm=bnorm)
        self.fn_loss = fn_loss.to("cuda:1")

        if not test:
            if optimzer == "adam":
                self.optim = torch.optim.Adam(self.model.parameters(), lr=kwargs["lr"], betas=(0.5, 0.999))
            
            self.st_epoch = 0
            self.device = device
            self.num_batch_train = kwargs["num_batch_train"]
            self.restart = restart
        
            self.writer_autoencoder = SummaryWriter(log_dir=kwargs["log_dir"])


    def train(self, loader_tr, loader_va, num_epochs):


        if self.restart:
            self.model, self.optim, self.st_epoch = load(ckpt_dir= os.path.join(self.ckpt_dir, "Auto_Encoder"), model=self.model, optim=self.optim)

        for epoch in range(self.st_epoch+1, num_epochs+1):

            train_loss = []
            validation_loss = []
            self.model.train()

            # Training
            for batch, data in enumerate(loader_tr, 1):

                img = data['label'].to("cuda:0")
                output = self.model(img)
                self.optim.zero_grad()

                loss = self.fn_loss(output, img.to("cuda:1"))
                train_loss += [loss.item()]
                loss.backward()

                self.optim.step()

                print(f"TRAIN : {epoch} / {num_epochs} | BATCH : {batch} / {self.num_batch_train} | LOSS : {np.mean(train_loss):.4f}")

                if batch % 500 == 0 or batch == self.num_batch_train:
                    output = self.to_numpy(output).squeeze()
                    output = np.clip(output, a_min=0, a_max=1)
                    id = self.num_batch_train * (epoch - 1)+ batch

                    self.writer_autoencoder.add_image("Input", img, id, dataformats="NCHW")    
                    self.writer_autoencoder.add_image("output", output, id, dataformats="NHWC")

            # Validating
            with torch.no_grad():
                self.model.eval()
                
                for batch, data in enumerate(loader_va, 1):
                    img = data['label'].to("cuda:0")
                    output = self.model(img)

                    loss = self.fn_loss(output, img.to("cuda:1"))
                    validation_loss += [loss.item()]
                
                print(f"Validation : {epoch} / {num_epochs} | LOSS : {np.mean(validation_loss):.4f}")

            self.writer_autoencoder.add_scalar("Train Loss", np.mean(train_loss), epoch)
            self.writer_autoencoder.add_scalar("Validation Loss", np.mean(validation_loss), epoch)
            save(ckpt_dir=self.ckpt_dir, model=self.model, optim=self.optim, epoch=epoch)

    def test(self, loader_te, num_batch_testing):

        test_loss = []
        
        self.model = load_eval(ckpt_dir= os.path.join(self.ckpt_dir, "Auto_Encoder"), model=self.model)

        with torch.no_grad():
            self.model.eval()

            for batch, data in enumerate(loader_te, 1):
                img = data['label'].to("cuda:0")
                output = self.model(img)

                loss = self.fn_loss(output, img.to("cuda:1"))
                test_loss += [loss.item()]

                print(f"Testing : {batch} / {num_batch_testing}")
                
            print(f"Testing.. | LOSS : {np.mean(test_loss):.4f}")
