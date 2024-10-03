import os

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as torchtransforms
from torch.utils.data import Dataset,ConcatDataset
from torch import optim
import torch

from .MRIimage import MRIImageSet_split_sys_HCP
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os


from volumentations import *
from torch.utils.data import DataLoader
def data_process(root_dir,crop_size=140,random_seed=42):
    """current_directory = os.path.dirname(__file__)
    root_slice=os.path.join(current_directory,root_dir,slice_dir)
    root_input=os.path.join(current_directory,root_dir,input_dir)"""
    """transform_set =transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
                    torchtransforms.RandomResizedCrop(
                        size=256,
                        scale=(1.0, 1.0),
                        ratio=(1.0, 1.0)
                    )
                ,
        transforms.ToTensor()])"""

    """ dataset_0=MRIImageSet_split_sys(root_dir=root_dir,transform=transform_set,crop_sz=crop_size,random_seed=random_seed)"""
    aug=get_augmentation((crop_size,crop_size,crop_size))
   
    dataset_1=MRIImageSet_split_sys_HCP(root_dir=root_dir,crop_sz=crop_size,transform=aug,random_seed=random_seed
        )
    #现在input文件夹内只有一个数据，就还没弄
    train_1,val_1=[],[]
    import pdb;
    for image,another_img,label,data_type in dataset_1:
        #pdb.set_trace()
        if data_type=="train":
            train_1.append([image,another_img,label])
        else:
            val_1.append([image,another_img,label])
    
   
    return train_1,val_1


def get_augmentation(patch_size):
    return Compose([
        Rotate(p=0.5),
        RandomCropFromBorders(crop_value=0.1, p=0.5),
        ElasticTransform( interpolation=2, p=0.1),
        Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
    ], p=1.0)

def save_dataloader_data(dataloader, file_name):
    # 初始化一个空列表，用来存放所有批次的数据
    batches = []
    
    # 迭代 DataLoader 并保存每个批次的数据
    for batch in dataloader:
        batches.append(batch)
    
    # 使用 torch.save 将所有批次的数据保存到文件
    torch.save(batches, file_name)
if __name__=="__main__":
    trainset,valset=data_process("F:\\HCP\\unzipnii")
    trainLoader=DataLoader(trainset,batch_size=16)
    valLoader=DataLoader(valset,batch_size=32)
    print(trainset)