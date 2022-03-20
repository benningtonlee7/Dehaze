from dataloader import *
import os
from glob import glob
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from pathlib import Path
from PIL import Image, ImageFile
import argparse
from model import *
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import datetime


def main():


    parser = argparse.ArgumentParser(description='Dehaze trainig')
    # Directories arguments
    parser.add_argument('--data-dir', type=str, metavar='<dir>', default="./train",
                        help='Directory to image dataset')


    # Optional arguments for training
    parser.add_argument('--resume', type=str, metavar='<.pth>', default=None,
                        help='File to save and load for resuming training, default=disabled')

    parser.add_argument('--save_dir', type=str, metavar='<dir>', default='./saved_models',
                        help='Directory to save trained models, default=./saved_models')
    parser.add_argument('--log_dir', type=str, metavar='<dir>', default='./logs',
                        help='Directory to save logs, default=./logs')
    parser.add_argument('--log_image_every', type=int, metavar='<int>', default=200,
                        help='Interval for logging dehazed images, negative for disabling, default=200')
    parser.add_argument('--save_interval', type=int, metavar='<int>', default=10000,
                        help='Interval for saving model, default=10000')
    parser.add_argument('--cuda', action='store_true', help='Option for using GPU if available')
    parser.add_argument('--n-worker', type=int, metavar='<int>', default=2,
                        help='Number of workers used for dataloader, default=2')

    # hyper-parameters
    parser.add_argument('--lr', type=float, metavar='<float>', default=1e-4,
                        help='Learning rate, default=1e-4')
    parser.add_argument('--learning_rate_decay', type=float, metavar='<float>', default=5e-5,
                        help='Learning rate decay, default=5e-5')
    parser.add_argument('--epoch', type=int, metavar='<int>', default=200,
                        help='Maximum number of epochs, default=200')
    parser.add_argument('--batch_size', type=int, metavar='<int>', default=8, help='Size of the batch, default=8')

    args = parser.parse_args()

    # Use GPU when it's available
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.is_available())
    haze_images_dir = os.path.join(args.data_dir, 'haze_images')
    haze_images_labels_dir = os.path.join(args.data_dir, 'haze_images_labels')
    clean_images = os.path.join(args.data_dir, 'clean_images')
    clean_images_labels = os.path.join(args.data_dir, 'clean_images_labels')

    # Image with name greater or equal to 170 will be considered as validation images.
    # Example 170.jpg, 173.jpg etc
    val_cutoff = 170

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Make directory for logs if not created already
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    # Training dataset
    train_dataset = ImageDataset(haze_images_dir, train_transform(), cut_off=val_cutoff)  # should be 159 of them
    val_dataset = ImageDataset(haze_images_dir, train_transform(), cut_off=val_cutoff, is_val=True)  # should be 18 of them

    print('Number of traing images :', len(train_dataset))
    print('Number of validation images :', len(val_dataset))

    # Dataloader
    train_iter = iter(DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker))
    val_iter = iter(DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_worker))

    model_dir = args.resume
    ckp = torch.load(model_dir, map_location=device)

    model = Dehaze()
    model = nn.DataParallel(model)
    model.load_state_dict(ckp['model'])
    model.eval()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if __name__ == "__main__":

    main()
