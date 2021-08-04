import argparse

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libs import *
from datasets import *
from torchvision import transforms


## Parser Setting
parser = argparse.ArgumentParser(description="Auto Encoder Trainer for DAPI images",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--data_dir", "-d", default="../Data/DAPI/GAN_DATA/good_case", type=str, dest="data_dir")                                 
parser.add_argument("--ckpt_dir", "-c", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", "-r", default="./result", type=str, dest="result_dir")
parser.add_argument("--name", "-n", default='auto_encoder', type=str, dest="name")

args = parser.parse_args()

data_dir      = args.data_dir
ckpt_dir      = os.path.join(args.ckpt_dir, "Auto_Encoder")
result_dir    = args.result_dir

name          = args.name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## print Parameters
print("==================================")
print(" Encoding Evaluation Information ")
print("==================================")
print(f"Name : {name}")
print(f"\nCheck Point Path : {ckpt_dir}")
print(f"Result Path : {result_dir}")
print(f"\nDevice : {device}")
print("==================================\n")


if __name__ == "__main__":
    
    
    model = AutoEncoder_De(in_channels=3, out_channels=3, nker=32, norm="bnorm")
    model = load_net(ckpt_dir=ckpt_dir, model=model, name=name)

    with torch.no_grad():
        model.eval()
        latent_vector = np.load(os.path.join(result_dir, "Train_Good_DAPI_Latente_Vector.npy"))
        latent_vector = torch.from_numpy(latent_vector).to("cuda:1")
        img = model(latent_vector).to("cpu").numpy().transpose(0, 2, 3, 1)
        img = np.clip(img, a_min=0, a_max=1)
        plt.imsave(os.path.join(result_dir, "Representation_DAPI.jpg"), img[0])

        
