import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import tqdm

from utils.models import Classifier
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy
from utils.dataset import NCaltech101


if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)

    print('Testing')
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    print('Successful')
