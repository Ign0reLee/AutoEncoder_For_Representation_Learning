import argparse

import os
from typing import no_type_check_decorator
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

parser.add_argument("--lr", "-lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", "-bs", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", "-ne", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", "-d", default="../Data/DAPI/GAN_DATA/good_case", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", "-c", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", "-l", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", "-r", default="./result", type=str, dest="result_dir")

parser.add_argument("--nker", "-n", default=32, type=int, dest="nker")
parser.add_argument("--restart", "-rs", default=False, type=bool, dest="restart")

args = parser.parse_args()

## Training Parameter Setting
learning_rate = args.lr
batch_size    = args.batch_size
num_epochs    = args.num_epoch

data_dir      = args.data_dir
ckpt_dir      = args.ckpt_dir
log_dir       = args.log_dir
result_dir    = args.result_dir
nker          = args.nker
restart       = args.restart

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## print Parameters
print("============================")
print("    Training Information    ")
print("============================")
if restart :print(f"Restarting..")
print(f"Rearning Late : {learning_rate}")
print(f"Batch Size : {batch_size}")
print(f"Number of Epochs :{num_epochs}")
print(f"\nData Path : {data_dir}")
print(f"Check Point Path :{ckpt_dir}")
print(f"Log Path : {log_dir}")
print(f"Result Path : {result_dir}")
print(f"\nDefualt Number Of Kener : {nker}")
print(f"\nDevice : {device} \n")


## If Directory is not exists, make it
if not os.path.exists(log_dir)    : os.mkdir(log_dir)
if not os.path.exists(result_dir) : os.mkdir(result_dir)
if not os.path.exists(ckpt_dir):
     os.mkdir(ckpt_dir)
     os.mkdir(os.path.join(ckpt_dir, "Encoder"))
     os.mkdir(os.path.join(ckpt_dir, "Decoder"))
     os.mkdir(os.path.join(ckpt_dir, "Auto_Encoder"))
      


if __name__ == "__main__":
    
    assert os.path.exists(data_dir), f"Please Input Right Data Path! Now path '{data_dir}' is not Exists!"
    transform_train = transforms.Compose([RandomFlip()])
    dataset_train = Dataset(data_dir=os.path.join(data_dir, "train"), transform=transform_train)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0) 
    dataset_valid = Dataset(data_dir=os.path.join(data_dir, "val"), transform=transform_train)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0)   
    num_data_train = len(dataset_train)
    num_batch_train = np.ceil(num_data_train / batch_size)

    AEtrainer = AETrainer(lr=learning_rate, num_batch_train=num_batch_train, ckpt_dir=ckpt_dir, log_dir=log_dir, restart=restart, nker=nker, device=device)
    AEtrainer.train(loader_tr=loader_train,loader_va=loader_valid, num_epochs=num_epochs)