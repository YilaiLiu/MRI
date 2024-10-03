import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from data.data_process import data_process
#from metrics import accuracy
from torch import nn
#from torchsummary import summary
import os
from data import mydatasets
from tqdm import tqdm
from models.my_model import Unet_convert
import loss
os.environ["WANDB_MODE"] = "offline"
import networks


Net=Unet_convert
root_dir="/groups/bachelor_research/home/share/DFNET/AD/ADungz"

params={          "device":"cuda",
                "lr":2e-4,
                "epoch":150,
                "weight_decay":0.01,
                "save_valset":True,
                "alpha":0.3}
project_name="ratio_sys"
name="stage1_firstTry"
# create model
import wandb
logger=wandb.init(
    project=project_name,
    name=name,
    #config= {**params_model, **params},
    config=params,
    resume=None

)
def set_requires_grad( nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
def save_checkpoint(state,epoch,save_dir="checkpoint"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name=f"epoch{epoch}"+"_"+str(state["val_loss"])+".pth"
    torch.save(state["model"],os.path.join(str(save_dir),file_name))

def train():
    output_dir='./result'
    #torch.manual_seed(0)
    checkpoint_dir,_=mydatasets.setup_logging(project_name+"_"+name)
    #这里后面多个rootdir记得混合数据
    os.makedirs(output_dir,exist_ok=True)
    output_dir=os.path.join(output_dir,project_name,name)
    trainset,valset=data_process(root_dir=root_dir,crop_size=60)
    device=params['device']
   
    
    trainLoader=DataLoader(trainset,batch_size=4)
    valLoader=DataLoader(valset,batch_size=1)

    if params["save_valset"]:
        save_dataloader_data(trainLoader,'trainset.pth')
        save_dataloader_data(valLoader, 'valset.pth')
    netG = Net(inch=1).to(params["device"])
    netD = networks.define_D(2, params.ndf, params.netD,
                                params.n_layers_D, params.norm, params.init_type, params.init_gain, 0)
    #############
    optimizer_G = torch.optim.Adam(netG, lr=params.lr, betas=(params.beta1, 0.999))
    optimizer_D = torch.optim.Adam(netD, lr=params.lr, betas=(params.beta1, 0.999))
    optimizers=[]
    optimizers.append(optimizer_G)
    optimizers.append(optimizer_D)
    valdata=iter(valLoader)
    ####################
    criterionGAN = networks.GANLoss(params.gan_mode).to(params.device)
    criterionL1 = torch.nn.L1Loss()
    for epoch in range(params['epoch']):
        netG.train()
        sum_loss=0
        step=0
        iter_data=tqdm(trainLoader)
        ############
        for i,(real_A,real_B,ratio) in enumerate(iter_data):
            #import pdb;pdb.set_trace()
            images=images.to(device).to(torch.float32)
            fake_B = netG(real_A)
            set_requires_grad(netD,True)
            optimizer_D.zero_grad()     # set D's gradients to zero
                   # calculate gradients for D
            fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = netD(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()          # update D's weights
            # update G
            set_requires_grad(netD, False)  # D requires no gradients when optimizing G
            optimizer_G.zero_grad()        # set G's gradients to zero
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = netD(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True)
            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B) * params.lambda_L1#默认为100
            # combine loss and calculate gradients
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()                      # calculate graidents for G
            optimizer_G.step()             # update G's weights
            step+=1
            mydatasets.save_image_nii(fake_B[0],os.path.join(output_dir,f'{epoch+1}.nii'))
    save_checkpoint({"G":netG.state_dict(),"D":netD.state_dict()},save_dir=checkpoint_dir)
            
        #iter_data.set_postfix({"loss":"{0:1.5f}".format(avg_loss)})
    """ print(f"train loss: {avg_loss}")
    with torch.no_grad():
        model.eval()
        try:
            images,another_imgs,labels=next(valdata)
        except:
            valdata=iter(valLoader)
            images,another_imgs,labels=next(valdata)
        images,another_imgs ,labels = images.unsqueeze(1).to(device),another_imgs.unsqueeze(1).to(device), labels.unsqueeze(1).to(device)
        output=model(images,another_imgs)
        output_sq=output.squeeze()
        loss_sum,ssim,L1=citerition(output_sq,labels).item()
        #############
        #acc=accuracy(pred,labels)
        print('validating! epoch: %d Loss: %.06f'
                    % (epoch+1, loss))
        logger.log({
            'train_loss': avg_loss,
            'val_loss': loss_sum,
            'ssim':ssim,
            'L1':L1,
            'epoch': epoch
        })
        mydatasets.save_image_nii(labels[0],os.path.join(output_dir,f'{epoch+1}.nii'))

        save_checkpoint({"model":model,"val_loss":loss},epoch=epoch,save_dir=checkpoint_dir)"""

def save_dataloader_data(dataloader, file_name):
    # 初始化一个空列表，用来存放所有批次的数据
    batches = []
    
    # 迭代 DataLoader 并保存每个批次的数据
    for batch in dataloader:
        batches.append(batch)
    
    # 使用 torch.save 将所有批次的数据保存到文件
    torch.save(batches, file_name)



if __name__=="__main__":
    train()
"""    trainset,valset=data_process(root_dir=root_dir)
    device="cuda:0"
    valLoader=DataLoader(valset,batch_size=32)
    data=iter(valLoader)
    
    model = Net(in_ch=1,num_classes=2)
    model.load_state_dict(torch.load("models\\binary-gibbs_third_exp\\epoch25.pt")["state_dict"])
    model=model.cuda()
    num=0
    acc_sum=0
    for images,labels in data:
        num+=1
        images, labels = images.to(device), labels.to(device)
        output=model(images)
        pred=torch.argmax(output,dim=-1)

        acc=accuracy(pred,labels)
        acc_sum+=acc
    print(acc_sum/num)"""
