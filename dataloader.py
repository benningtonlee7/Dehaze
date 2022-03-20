import numpy as np
from torch.utils.data import Dataset, sampler
from torchvision import transforms
from PIL import Image
import glob
from pathlib import Path
import cv2
import os


class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None, cut_off=None, is_val=False):

        super(ImageDataset, self).__init__()
        # root directory
        self.root_dir = root_dir
        # get a list of image names in file directory
        self.img_files = glob.glob(self.root_dir+'/*.jpg')
        if cut_off is not None:
            tmp = []
            for i in range(len(self.img_files)):
                if is_val:
                    if int(Path(self.img_files[i]).stem) >= cut_off:
                        tmp.append(Path(self.img_files[i]))
                else:
                    if int(Path(self.img_files[i]).stem) < cut_off:
                        tmp.append(Path(self.img_files[i]))
            self.img_files = tmp

        self.hazy_imgs = []
        self.clean_imgs = []
        # Split hazed and clean images
        for img in self.img_files:
            hazy_img, clean_img = split_img("", img)
            self.hazy_imgs.append(hazy_img)
            self.clean_imgs.append(clean_img)

        self.transform = transform

    def __getitem__(self, index):
        """
        Load an input image at this index from the root, convert it to the format VGG
        accepts.

        return: img tensor
        """

        hazed_img = self.hazy_imgs[index]
        clean_img = self.clean_imgs[index]

        if self.transform:
            hazed_img = self.transform(hazed_img)
            clean_img = self.transform(clean_img)

        return hazed_img, clean_img

    def __len__(self):
        return len(self.hazy_imgs)


# transform for the training set
# augment training dataset with randomly rotated by 90, 180, 270
#
# degrees and horizontal flip.
def train_transform(size=None):
    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.RandomCrop([256, 256]))
    transform_list.append(transforms.RandomRotation([90, 270]))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


# transform for the testing set
def test_transform(size=None, crop=False):
    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    # add normalization
    transform_list.append(transforms.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152]))
    return transforms.Compose(transform_list)


# Infinite iterator
def InfiniteSamplerIterator(n):

    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

def split_img(root_dir, img_name):
    '''
    The image comprises of hazy image and clean image
    The top half is the hazy image and the bottom half is clean image

      Parameters:
              root_dir: Path to the image
              Img_name:  Image file neam

      Returns:
              The hazy image as a numpy array
    '''
    img = cv2.imread(os.path.join(root_dir, img_name))
    hazed_img = img[:int(img.shape[0] / 2)][:, :, ::-1]  # BGR -> RGB
    clean_img = img[int(img.shape[0] / 2):][:, :, ::-1]
    return hazed_img, clean_img

# Infinite random sampler which implement InfiniteSamplerIterator
class InfiniteSampler(sampler.Sampler):

    def __init__(self, num):
        self.num_samples = num

    def __iter__(self):
        return iter(InfiniteSamplerIterator(self.num_samples))

    def __len__(self):
        return 2 ** 31
