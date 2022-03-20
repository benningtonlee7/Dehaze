import argparse
import copy
import logging
from datetime import datetime
from model import *
from PIL import Image

from dataloader import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as tfs 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--model-dir', type=str, metavar='<.pth>', default=None,
                        help='File to load for evaluation, default=disabled')
    parser.add_argument('--cuda', action='store_true', help='Option for using GPU if available')
    parser.add_argument('--data-dir', default='./train', type=str)
    parser.add_argument('--log-dir', type=str, metavar='<dir>', default='./logs',
                        help='Directory to save logs, default=./logs')
    parser.add_argument('--output-dir', type=str, metavar='<dir>', default='./results',
                        help='Directory to save results, default=./results')
    parser.add_argument('--n-worker', type=int, metavar='<int>', default=2,
                        help='Number of workers used for dataloader, default=2')
    return parser.parse_args()


def evaluate(model, img_dir, output_dir):

    model.eval()
    test_tf = test_transform()
    with torch.no_grad():

        total_time = datetime.now()

        for img in os.listdir(img_dir):
            start_time = datetime.now()
            hazy_img, clean_img = split_img(img_dir, img)
            npad = ((0, 0), (0, 4), (0, 0))
            hazy_img = np.pad(hazy_img, npad, 'constant')
            clean_img = np.pad(clean_img, npad, 'constant')
            hazy_img = test_tf(hazy_img)[None, ::]  # Give it an additional dimension
            clean_img = test_tf(clean_img)[None, ::]
            #hazy_img = Image.open(img_dir + img)
            #hazy_img = tfs.Compose([
            #             tfs.ToTensor(),
            #             tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
            #          ])(hazy_img)[None, ::]
            output, _ = model(hazy_img)
                    
            end_time = datetime.now()
            total_time += (end_time - start_time)

            res = output.cpu()
            vutils.save_image(res, str(output_dir) + "/" + str(Path(img).stem) + '.jpg')
            print("Img {} is done. Time it took: {}".format(Path(img).stem, (end_time-start_time)))
    return total_time


def main():

    args = get_args()

    logger = logging.getLogger(__name__)
    logger.info(args)
    print(args)
    haze_images_dir = os.path.join(args.data_dir, 'haze_images')
    haze_images_labels_dir = os.path.join(args.data_dir, 'haze_images_labels')
    clean_images = os.path.join(args.data_dir, 'clean_images')
    clean_images_labels_dir = os.path.join(args.data_dir, 'clean_images_labels')

    val_cutoff = 170
    val_dataset = ImageDataset(haze_images_dir, test_transform(), cut_off=val_cutoff, is_val=True)  # should be 18 of them
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_worker)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    ckp = torch.load(args.model_dir, map_location=device)
    model = Dehaze()
    model = nn.DataParallel(model)
    model.load_state_dict(ckp['model'])
    total_time = evaluate(model, img_dir=haze_images_dir, output_dir=output_dir)
    print("Runtime: {}".format(total_time))


if __name__ == "__main__":
    main()







