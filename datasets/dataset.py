import os
import torch
import torch.utils.data
from PIL import Image
from datasets.data_augment import PairCompose, PairToTensor, PairRandomHorizontalFilp
import numpy as np
import imageio
import random

def remove_black_level(img, black_level=63, white_level=4*255):
    img = np.maximum(img-black_level, 0) / (white_level-black_level)
    return img

class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = AllWeatherDataset(self.config.data.data_dir,
                                          patch_size=self.config.data.patch_size,
                                          filelist='train.txt')
        val_dataset = AllWeatherDataset(self.config.data.data_dir,
                                        patch_size=self.config.data.patch_size,
                                        filelist='val.txt', train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        img_exts = (".png", ".jpg", ".npy", ".tiff", ".tif")

        def get_paths(dir_path):
            if not os.path.exists(dir_path):
                return []
            return [
                os.path.join(dir_path, fname)
                for fname in os.listdir(dir_path)
                if fname.endswith(img_exts)
            ]

        self.dir = dir
        self.train = train
        self.patch_size = patch_size
        if 'zrr' in self.dir:
            self.black_level, self.white_level, self.rho = 63, 1020, 8
        elif 'mai' in self.dir:
            self.black_level, self.white_level, self.rho = 255, 4095, 8
        else:
            raise ValueError('Get wrong dataset name:{}'.format(dir))

        self.input_paths = sorted(get_paths(os.path.join(dir, 'train/source')))
        self.gt_paths = sorted(get_paths(os.path.join(dir, 'train/target')))
        if not train:
            self.input_paths = sorted(get_paths(os.path.join(dir, 'test/source')))
            self.gt_paths = sorted(get_paths(os.path.join(dir, 'test/target')))



    def random_crop(self, input_img, gt_img, gt_img_gray):
        _, h,w = input_img.shape
        x0 = random.randrange(start=0,stop=h-self.patch_size,step=2)
        y0 = random.randrange(start=0,stop=w-self.patch_size, step=2)

        input_img_crop = input_img[:,x0:x0+self.patch_size, y0:y0+self.patch_size]

        gt_img_crop = gt_img[:,x0:x0+self.patch_size, y0:y0+self.patch_size]
        gt_img_gray_crop = gt_img_gray[:,x0:x0+self.patch_size, y0:y0+self.patch_size]

        return input_img_crop, gt_img_crop, gt_img_gray_crop
    

    def get_images(self, index):
        input_name = self.input_paths[index]
        gt_name = self.gt_paths[index]
        img_id = input_name.split('/')[-1].split('.')[0]
        
        input_img = np.expand_dims(np.asarray(np.asarray(imageio.imread(input_name))), axis=0)
        gt_img = np.expand_dims(np.asarray(np.asarray(imageio.imread(gt_name))), axis=0)
        
        input_raw, gt_img, gt_img_gray = np.expand_dims(np.asarray(imageio.imread(input_name)), axis=-1), \
            np.asarray(imageio.imread(gt_name)), \
            np.expand_dims(np.asarray(imageio.imread(gt_name, mode='L')), axis=-1)
        

        input_raw = np.maximum((input_raw - self.black_level), 0) / (self.white_level - self.black_level)
        # input_raw = np.clip(input_raw, 0, 1)
        # input_raw = np.maximum((input_raw-np.min(input_raw)), 0) / (np.max(input_raw) - np.min(input_raw))
        # input_raw = input_raw / np.max(input_raw)
        input_raw, gt_img, gt_img_gray = torch.tensor(input_raw).permute(2, 0, 1).float(), \
            torch.tensor(gt_img / 255.0).permute(2, 0, 1).float(), \
            torch.tensor(gt_img_gray / 255.0).permute(2, 0, 1).float()
        if self.train:
            input_raw, gt_img, gt_img_gray = \
                self.random_crop(input_raw, gt_img, gt_img_gray)
            # return input_raw, gt_img, gt_img_gray

            # unwarped_gt_name = name.
            # unwarped_gt_img = np.asarray(imageio.imread(unwarped_gt_name))
            # unwarped_gt_img = torch.tensor(unwarped_gt_img/255.0).permute(2, 0, 1).float() 
        return input_raw, gt_img, gt_img_gray,img_id
        


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_paths)
