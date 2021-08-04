import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libs import *
from datasets import *
from torchvision import transforms

## Parser Setting
parser = argparse.ArgumentParser(description="Auto Encoder Trainer for DAPI images",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--batch_size", "-bs", default=1, type=int, dest="batch_size")
parser.add_argument("--data_dir", "-d", default="../Data/DAPI/GAN_DATA/good_case", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", "-c", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", "-l", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", "-r", default="./result", type=str, dest="result_dir")

parser.add_argument("--nker", "-n", default=32, type=int, dest="nker")

args = parser.parse_args()

batch_size    = args.batch_size
data_dir      = args.data_dir
ckpt_dir      = args.ckpt_dir
log_dir       = args.log_dir
result_dir    = args.result_dir
nker          = args.nker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## print Parameters
print("============================")
print("    Training Information    ")
print("============================")
print(f"Data Path : {data_dir}")
print(f"Check Point Path :{ckpt_dir}")
print(f"Log Path : {log_dir}")
print(f"Result Path : {result_dir}")
print(f"\nBatch Size : {batch_size}")
print(f"\nDefualt Number Of Kener : {nker}")
print(f"\nDevice : {device} \n")



if __name__ == "__main__":
    
    assert os.path.exists(data_dir), f"Please Input Right Data Path! Now path '{data_dir}' is not Exists!"

    dataset_test = Dataset(data_dir=os.path.join(data_dir, "test"), transform=None)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0) 
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    AEtrainer = AETrainer(ckpt_dir=ckpt_dir, log_dir=log_dir, nker=nker, device=device, test=True)
    AEtrainer.test(loader_te=loader_test, num_batch_testing=num_batch_test)
