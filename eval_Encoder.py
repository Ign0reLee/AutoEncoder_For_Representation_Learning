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


parser.add_argument("--batch_size", "-bs", default=4, type=int, dest="batch_size")
parser.add_argument("--data_dir", "-d", default="../Data/DAPI/GAN_DATA/good_case", type=str, dest="data_dir")                                 
parser.add_argument("--ckpt_dir", "-c", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", "-r", default="./result", type=str, dest="result_dir")
parser.add_argument("--name", "-n", default='auto_encoder', type=str, dest="name")

args = parser.parse_args()

batch_size    = args.batch_size
data_dir      = args.data_dir
ckpt_dir      = os.path.join(args.ckpt_dir, "Auto_Encoder")
# ckpt_dir      = os.path.join(args.ckpt_dir, "Auto_Encoder")
result_dir    = args.result_dir

name          = args.name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## print Parameters
print("==================================")
print(" Encoding Evaluation Information ")
print("==================================")
print(f"Name : {name}")
print(f"Batch Size : {batch_size}")
print(f"\nData Path : {data_dir}")
print(f"Check Point Path : {ckpt_dir}")
print(f"Result Path : {result_dir}")
print(f"\nDevice : {device}")
print("==================================\n")


if __name__ == "__main__":
    
    assert os.path.exists(data_dir), f"Please Input Right Data Path! Now path '{data_dir}' is not Exists!"
    dataset_train = Dataset(data_dir=os.path.join(data_dir, "train"))
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    num_data_train = len(dataset_train)
    num_batch_train = np.ceil(num_data_train / batch_size)

    # model = netE(in_channels=3, nker=32).to(device)
    model = AutoEncoder_EN(in_channels=3, out_channels=3, nker=32, norm="bnorm").to(device)
    model = load_net(ckpt_dir=ckpt_dir, model=model, name=name)

    with torch.no_grad():
        model.eval()
        vector_sum = torch.zeros((1, 32 * 32, 15, 20)).to(device) # latent vector size
        for batch, data in enumerate(loader_train, 1):
            img = data['label'].to(device)
            latent_vector = model(img)
            latent_vector = torch.mean(latent_vector, dim=0)
            vector_sum = torch.add(vector_sum, latent_vector)
            print(f"BATCH : {batch} / {num_batch_train}")
        
        vector_sum = vector_sum / batch
        vector_sum = vector_sum.to("cpu").numpy()
        np.save(os.path.join(result_dir, "Train_Good_DAPI_Latente_Vector.npy"),vector_sum)
        
