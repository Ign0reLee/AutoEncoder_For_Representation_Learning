import torch
import torch.nn as nn

import os
import numpy as np


def save(ckpt_dir, model, optim, epoch):
    parameterE = model.netE
    parameterD = model.netD

    torch.save({"auto_encoder": model.state_dict(), "optimizer" : optim.state_dict()},os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth"))
    torch.save({"encoder" : parameterE.state_dict()}, os.path.join(ckpt_dir, "Encoder", f"model_epoch_{epoch}.pth"))
    torch.save({"encoder" : parameterD.state_dict()}, os.path.join(ckpt_dir, "Decoder", f"model_epoch_{epoch}.pth"))


def load(ckpt_dir, model, optim):

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    model.load_state_dict(dict_model['auto_encoder'])
    optim.load_state_dict(dict_model["optimizer"])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return model, optim, epoch