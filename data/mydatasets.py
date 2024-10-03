import os
import torchvision
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
def plot_image(images):
    #channel*h*w
    plt.figure(figsize=(32,32))
    figures=torch.cat(torch.cat([i for i in images.cpu],dim=-1),dim=-2).permute(1,2,0).cpu()
    #plt需要h*w*channel
    plt.imshow(figures)
    plt.show()

def save_image(images,path,**kwargs):
    grid=torchvision.utils.make_grid(images,**kwargs)
    ndarr=grid.permute(1,2,0).to("cpu").numpy()
    im=Image.fromarray(ndarr)
    im.save(path)

def save_image_nii(image,path,**kwargs):
    image=image.numpy()
    nii=nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(nii,path)

def setup_logging(run_name):
    os.makedirs("models",exist_ok=True)
    os.makedirs("result",exist_ok=True)
    checkpoints_dir=os.path.join("models",run_name)
    os.makedirs(checkpoints_dir,exist_ok=True)
    output_dir=os.path.join("result",run_name)
    os.makedirs(output_dir,exist_ok=True)
    return checkpoints_dir,output_dir