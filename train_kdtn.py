from utils.dataloader import *
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
from models.model import *
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import time
from utils.metrics import psnr, ssim


def lr_schedule_cosdecay(t, T, init_lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(model, train_dataloader, test_dataloader, optimizer, criterion, args, device):
    losses = []
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    epoch_size = len(train_dataloader)
    # continue training from last time if resume
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch - 1}.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch - 1}.pth')))
        print(f'Resuming at epoch {start_epoch}')
    else:
        start_epoch = 0
        print('training from scratch')
    for epoch in range(start_epoch, args.epoch):
        just_saved = False
        start_time = time.time()
        model.train()
        epoch_loss = 0
        for step, (hazy_img, clean_img) in enumerate(train_dataloader):
            hazy_img = hazy_img.to(device)
            clean_img = clean_img.to(device)
            out, _ = model(hazy_img)
            out = out.to(device)
            loss = criterion(out, clean_img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            epoch_loss += loss.item()

        print(
            f'Training loss:{epoch_loss:.5f} |Epoch time_used :{(time.time() - start_time):.1f}')

        model.eval()
        for step, (hazy_img, clean_img) in enumerate(test_dataloader):
            hazy_img = hazy_img.to(device)
            clean_img = clean_img.to(device)
            with torch.no_grad():
                out, _ = model(hazy_img)
            out = out.to(device)
            ssim1 = ssim(out, clean_img).item()
            psnr1 = psnr(out, clean_img)
            ssims.append(ssim1)
            psnrs.append(psnr1)
        ssim_eval, psnr_eval = np.mean(ssims), np.mean(psnrs)

        # this metrics can be swaped
        if psnr_eval > max_psnr:
            max_ssim = max(max_ssim, ssim_eval)
            max_psnr = max(max_psnr, psnr_eval)
            save_model_dir = os.path.join(args.save_dir, 'model_best.pth')
            print(
                f'model saved at epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')

            torch.save({
                'epoch': epoch,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_model_dir)
            torch.save(optimizer.state_dict(), os.path.join(args.save_dir, f'opt_best.pth'))
            just_saved = True

        if (epoch + 1) % 10 == 0 and not just_saved:
            print(
                f'model saved at epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
            save_model_dir = os.path.join(args.save_dir, f'model_{epoch}.pth')
            torch.save(optimizer.state_dict(), os.path.join(args.save_dir, f'opt_{epoch}.pth'))
            torch.save({
                'epoch': epoch,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_model_dir)

        print(f'epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

def main():
    parser = argparse.ArgumentParser(description='Dehaze trainig')
    # Directories arguments
    parser.add_argument('--data-dir', type=str, metavar='<dir>', default="./train",
                        help='Directory to image dataset')

    # Optional arguments for training
    parser.add_argument('--resume', type=str, metavar='<int>', default=None,
                        help='Resume epoch for resuming training, default=disabled')
    parser.add_argument('--saved_model', type=str, metavar='<.pth>', default=None,
                        help='File to load for resuming training, default=disabled')

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
    parser.add_argument('--lr_decay', type=float, metavar='<float>', default=5e-5,
                        help='Learning rate decay, default=5e-5')
    parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')

    parser.add_argument('--epoch', type=int, metavar='<int>', default=200,
                        help='Maximum number of epochs, default=100')
    parser.add_argument('--batch_size', type=int, metavar='<int>', default=16, help='Size of the batch, default=8')

    args = parser.parse_args()

    # Use GPU when it's available
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

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
    train_dataset = ImageDataset(haze_images_dir,transform=train_transform(), cut_off=val_cutoff)  # should be 159 of them
    val_dataset = ImageDataset(haze_images_dir, transform=train_transform(), cut_off=val_cutoff,
                             is_val=True)  # should be 18 of them

    print('Number of traing images:', len(train_dataset))
    print('Number of validation images:', len(val_dataset))

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_worker)
    
    model_dir = args.saved_model
    ckp = torch.load(model_dir, map_location=device)   # loading a pretrained and preprocess model
    model = Dehaze()
    model = nn.DataParallel(model)
    model.load_state_dict(ckp)

    optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                  betas=(0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()
    criterion = nn.L1Loss().to(device)  # reconstruction loss; other alternatives could be better

    train(model, train_dataloader, val_dataloader, optimizer, criterion, args, device)

if __name__ == "__main__":
    main()
