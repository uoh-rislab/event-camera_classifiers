from os.path import dirname
import argparse
import torch
import tqdm
import os

from utils.loader import Loader
from utils.models import Classifier
from utils.loss import cross_entropy_loss_and_accuracy
from utils.dataset import NCaltech101

import numpy as np

def FLAGS():
    parser = argparse.ArgumentParser(
        """Deep Learning for Events. Supply a config file.""")

    # can be set in config
    parser.add_argument("--checkpoint", default="", required=True)
    #parser.add_argument("--test_dataset", default="", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)

    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.checkpoint)), f"Checkpoint{flags.checkpoint} not found."
    #assert os.path.isdir(flags.test_dataset), f"Test dataset directory {flags.test_dataset} not found."

    print(f"----------------------------\n"
          f"Starting testing with \n"
          f"checkpoint: {flags.checkpoint}\n"
          #f"test_dataset: {flags.test_dataset}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"----------------------------")

    return flags


if __name__ == '__main__':
    flags = FLAGS()

    # model, load and put to device
    model = Classifier()
    #ckpt = torch.load(flags.checkpoint)
    #model.load_state_dict(ckpt["state_dict"])
    model = model.to(flags.device)

    model = model.eval()
    
    #paths = ['clean_100ms/training/']
    
    base_path = '/home/ignacio/Proyectos/UOH/GAN/AffWild/db_expr/'

    paths = ['Train_Set/', 'Validation_Set/']

    paths = [base_path + x for x in paths]

    ####print(paths)

    for path in paths:

        for i in range(0,7):

            files = os.listdir(path + str(i) + '/numpy/')
            files.sort()

            for file in files:
                if (file.endswith(".npy")):

                    print(file)

                    filename = file[:-4]

                    img_path = path + str(i) + '/numpy/' + filename + '.jpg'
                    img_path = img_path.replace('numpy', 'est')
                    npy_path = path + str(i) + '/numpy/' + file

                    print(npy_path)
                    ####print(img_path)


                    np_array = np.load(npy_path)

                    b = np.zeros((np_array.shape[0], np_array.shape[1] + 1 ))
                    b[:,:-1] = np_array
                    np_array = b

                    device = torch.device('cuda:0')
                    x_np = torch.from_numpy(np_array).float().to(device)

                    model(x_np, img_path)
            

