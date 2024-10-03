import os
import nibabel as nib
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as torchtransforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import random
class MRIImageSet(Dataset):
    """ImageNet dataset.

    Args:
        root_dir (str): Path to ImageNet root directory.
        crop_sz (int): Size to crop loaded images to (passed to 
            torchtransforms.RandomResizedCrop).
        transform (list, default=None): List of transforms to apply to loaded
            images.
    """

    def __init__(self, label,root_dir,crop_sz=256, transform=None):
 
        self.root_dir = root_dir
        self.file_list=[os.path.join(root_dir,file) for file in os.listdir(root_dir) if file.endswith("png")]
        self.dat_norm_fact = 255  # actually 189.1
        self.label=np.array(label)
        self.crop_sz=crop_sz
        self.transform=transform


    def __len__(self):
        num = len(self.file_list)
        return num
   

    def __getitem__(self, index):
        y = Image.open(self.file_list[index])
        y=y.convert("L")
        x0=(y.size[-2]-self.crop_sz)//2
        y0=(y.size[-1]-self.crop_sz)//2
        y.crop((x0,y0,x0+self.crop_sz,y0+self.crop_sz))   
        if self.transform !=None:
            y=self.transform(y)
        #entropy requirement
        return y,torch.tensor(self.label,dtype=torch.long)



class MRIImageSet_split(Dataset):
    """ImageNet dataset.

    Args:
        root_dir (str): Path to ImageNet root directory.
        crop_sz (int): Size to crop loaded images to (passed to 
            torchtransforms.RandomResizedCrop).
        transform (list, default=None): List of transforms to apply to loaded
            images.
    """

    def __init__(self, label,root_dir,crop_sz=256, transform=None,index_symbol="SMR",index_MAX=3,train_size=0.8,random_seed=None):
 
        self.root_dir = root_dir
        self.file_list=[file for file in os.listdir(root_dir) if file.endswith("png")]
        self.dat_norm_fact = 255  # actually 189.1
        self.label=np.array(label)
        self.crop_sz=crop_sz
        self.transform=transform
        self.index_MAX=index_MAX
        assert train_size<1 and train_size>0,"train_size should be in the range [0,1]"
        self.random_seed=random_seed
        self.train_size=train_size
        self.index_symbol=index_symbol
        self.train_list,self.val_list=self.train_test_spilt()
        print(self.train_list)
        import pdb
       #pdb.set_trace()
        
    def __len__(self):
        num = len(self.file_list)
        return num
    
    def train_test_spilt(self):
        SMR_list=[]
        seed=self.random_seed
        if seed!=None:
            random.seed(seed)
        #get list
        for file_name in self.file_list:
            index=file_name.find(self.index_symbol)
            if index!=-1:
                num=file_name[index+3:index+3+self.index_MAX]
                if num not in SMR_list:
                    SMR_list.append(num)
        length=len(SMR_list)
        print(SMR_list)
        train_length=round(length*self.train_size)
        print(f"{length} groups are divided by {self.index_symbol} ")
        train_list=random.sample(SMR_list,train_length)
        val_list=[elem for elem in SMR_list if elem not in train_list]
        return train_list,val_list
    def __getitem__(self, index):
        

        
        file_name=self.file_list[index]
        sub=file_name[:file_name.find("_")] if self.label==1 else 0
        y = Image.open(os.path.join(self.root_dir,file_name))
        y=y.convert("L")
        x0=(y.size[-2]-self.crop_sz)//2
        y0=(y.size[-1]-self.crop_sz)//2
        y.crop((x0,y0,x0+self.crop_sz,y0+self.crop_sz))   
        if self.transform !=None:
            y=self.transform(y)


        index=file_name.find(self.index_symbol)
        if index!=-1:
            num=file_name[index+3:index+3+self.index_MAX]
        if num in self.train_list:
            return y,torch.tensor(self.label,dtype=torch.long),int(sub),"train"
        else:
            return y,torch.tensor(self.label,dtype=torch.long),int(sub),"val"
        
    

class MRIImageSet_split_sys(Dataset):
    """ImageNet dataset.

    Args:
        root_dir (str): Path to ImageNet root directory.
        crop_sz (int): Size to crop loaded images to (passed to 
            torchtransforms.RandomResizedCrop).
        transform (list, default=None): List of transforms to apply to loaded
            images.
    """
    #T1/T2图像在和T1图像平级的一个文件夹中，对应的名字是label_file

    def __init__(self,root_dir,crop_sz=256, transform=None,input_type="T1",label_file="wemt1w_on_t2w_masked.nii",index_symbol="SMR",index_MAX=3,train_size=0.8,random_seed=None):
        assert input_type=="T1" or "T2","the type of input image should be T1 or T2"
        other_type="T1" if input_type=="T2" else "T2" 
        self.root_dir = root_dir
        self.file_list=[]
        self.SMRfolder=[]
        self.other_file_list=[]
        for SMRfolder in os.listdir(root_dir):
            self.SMRfolder.append(SMRfolder)
            self.file_list.extend(os.path.join(SMRfolder,file) for file in os.listdir(os.path.join(root_dir,SMRfolder)) if file.endswith(".nii") and input_type in file)
            self.other_file_list.extend(os.path.join(SMRfolder,file) for file in os.listdir(os.path.join(root_dir,SMRfolder)) if file.endswith(".nii") and other_type in file)
        #self.file_list=[file for file in os.listdir(root_dir) if file.endswith(".nii") and input_type in file]
        self.dat_norm_fact = 255 
        self.crop_sz=crop_sz
        self.transform=transform
        self.index_MAX=index_MAX#维度的最大位数
        assert train_size<1 and train_size>0,"train_size should be in the range [0,1]"
        self.random_seed=random_seed
        self.train_size=train_size
        self.index_symbol=index_symbol
        self.train_list,self.val_list=self.train_test_spilt()
        self.label_file=label_file
        print(self.train_list)
        import pdb
       #pdb.set_trace()
        
    def __len__(self):
        num = len(self.file_list)
        return num
    
    def train_test_spilt(self):
        SMR_list=[]
        seed=self.random_seed
        if seed!=None:
            random.seed(seed)
        #get list
        for num in self.SMRfolder:
            """index=file_name.find(self.index_symbol)
            if index!=-1:
                num=file_name[index+3:index+3+self.index_MAX]"""
            if num not in SMR_list:
                SMR_list.append(num)
        length=len(SMR_list)
        print(SMR_list)
        train_length=round(length*self.train_size)
        print(f"{length} groups are divided by {self.index_symbol} ")
        train_list=random.sample(SMR_list,train_length)
        val_list=[elem for elem in SMR_list if elem not in train_list]
        return train_list,val_list
    def __getitem__(self, index):
        file_name=self.file_list[index]
        #sub=file_name[:file_name.find("_")] if self.label==1 else 0
        #y = Image.open(os.path.join(self.root_dir,file_name))
        #pdb.set_trace()
        img_path=os.path.join(self.root_dir,file_name)
        img_nib = nib.load(img_path)
        anotherfile_name=self.other_file_list[index]
        anotherimg_path=os.path.join(self.root_dir,anotherfile_name)
        anotherimg_nib = nib.load(anotherimg_path)
        anothery=anotherimg_nib.get_fdata()



        y = img_nib.get_fdata() 
        sys_path=os.path.join(img_path.split(".")[0],self.label_file)
        sys_nib=nib.load(sys_path)
        sys = sys_nib.get_fdata() 
        #pdb.set_trace()
        """x0=(y.size[-2]-self.crop_sz)//2
        y0=(y.size[-1]-self.crop_sz)//2
        z0=(y.size[-3]-self.crop_sz)//2
        y.crop((x0,y0,z0,x0+self.crop_sz,y0+self.crop_sz,z0+self.crop_sz))   
        x1=(sys.size[-2]-self.crop_sz)//2
        y1=(sys.size[-1]-self.crop_sz)//2
        z1=(sys.size[-3]-self.crop_sz)//2
        sys.crop((x1,y1,z1,x1+self.crop_sz,y1+self.crop_sz,z1+self.crop_sz))  """
        #pdb.set_trace()
        if self.transform !=None:
            seed=np.random.randint(1,1000)
            random.seed(seed) 
            y=self.transform(image=y)['image']
            random.seed(seed) 
            anothery=self.transform(image=anothery)['image']
            random.seed(seed) 
            sys=self.transform(image=sys)['image']
            random.seed(None)
        num=os.path.dirname(file_name)
        if num in self.train_list:
            return y,anothery,sys,"train"
        else:
            return y,anothery,sys,"val"
        






  

class MRIImageSet_split_sys_HCP(Dataset):
    """ImageNet dataset.

    Args:
        root_dir (str): Path to ImageNet root directory.
        crop_sz (int): Size to crop loaded images to (passed to 
            torchtransforms.RandomResizedCrop).
        transform (list, default=None): List of transforms to apply to loaded
            images.
    """
    #T1/T2图像在和T1图像平级的一个文件夹中，对应的名字是label_file

    def __init__(self,root_dir,crop_sz=256, transform=None,input_type="T1w_acpc",index_symbol="SMR",index_MAX=3,train_size=0.8,random_seed=None,ratio_name="T1wDividedByT2w_ribbon.nii.gz"):
        assert input_type=="T1w_acpc" or "T2w_acpc","the type of input image should be T1 or T2"
        other_type="T1w_acpc" if input_type=="T2w_acpc" else "T2w_acpc" 
        self.root_dir = root_dir
        self.ratio_name=ratio_name
        self.file_list=[]
        self.SMRfolder=[]
        self.other_file_list=[]
        for SMRfolder in os.listdir(root_dir):
            self.SMRfolder.append(SMRfolder)
            self.file_list.extend(os.path.join(SMRfolder,"T1w",file) for file in os.listdir(os.path.join(root_dir,SMRfolder,"T1w")) if file.endswith(".nii") and input_type in file)
            self.other_file_list.extend(os.path.join(SMRfolder,"T1w",file) for file in os.listdir(os.path.join(root_dir,SMRfolder,"T1w")) if file.endswith(".nii") and other_type in file)
            self.other_file_list.extend(os.path.join(SMRfolder,"T1w",file) for file in os.listdir(os.path.join(root_dir,SMRfolder,"T1w")) if file.endswith(".nii.gz") and file==ratio_name)
        #self.file_list=[file for file in os.listdir(root_dir) if file.endswith(".nii") and input_type in file]
        self.dat_norm_fact = 255 
        self.crop_sz=crop_sz
        self.transform=transform
        self.index_MAX=index_MAX#维度的最大位数
        assert train_size<1 and train_size>0,"train_size should be in the range [0,1]"
        self.random_seed=random_seed
        self.train_size=train_size
        self.index_symbol=index_symbol
        self.train_list,self.val_list=self.train_test_spilt()
    
        print(self.train_list)
        """ import pdb
        pdb.set_trace()"""
        
    def __len__(self):
        num = len(self.file_list)
        return num
    
    def train_test_spilt(self):
        SMR_list=[]
        seed=self.random_seed
        if seed!=None:
            random.seed(seed)
        #get list
        for num in self.SMRfolder:
            """index=file_name.find(self.index_symbol)
            if index!=-1:
                num=file_name[index+3:index+3+self.index_MAX]"""
            if num not in SMR_list:
                SMR_list.append(num)
        length=len(SMR_list)
        print(SMR_list)
        train_length=round(length*self.train_size)
        print(f"{length} groups are divided by {self.index_symbol} ")
        train_list=random.sample(SMR_list,train_length)
        val_list=[elem for elem in SMR_list if elem not in train_list]
        return train_list,val_list
    def __getitem__(self, index):
        """import pdb;pdb.set_trace()"""
        file_name=self.file_list[index]
        #sub=file_name[:file_name.find("_")] if self.label==1 else 0
        #y = Image.open(os.path.join(self.root_dir,file_name))
        #pdb.set_trace()
        img_path=os.path.join(self.root_dir,file_name)
        img_nib = nib.load(img_path)
        anotherfile_name=self.other_file_list[index]
        anotherimg_path=os.path.join(self.root_dir,anotherfile_name)
        anotherimg_nib = nib.load(anotherimg_path)
        anothery=anotherimg_nib.get_fdata()



        y = img_nib.get_fdata() 
        sys_path=os.path.join(os.path.dirname(img_path),self.ratio_name)
        sys_nib=nib.load(sys_path)
        sys = sys_nib.get_fdata() 
       
        """x0=(y.size[-2]-self.crop_sz)//2
        y0=(y.size[-1]-self.crop_sz)//2
        z0=(y.size[-3]-self.crop_sz)//2
        y.crop((x0,y0,z0,x0+self.crop_sz,y0+self.crop_sz,z0+self.crop_sz))   
        x1=(sys.size[-2]-self.crop_sz)//2
        y1=(sys.size[-1]-self.crop_sz)//2
        z1=(sys.size[-3]-self.crop_sz)//2
        sys.crop((x1,y1,z1,x1+self.crop_sz,y1+self.crop_sz,z1+self.crop_sz))  """
        #pdb.set_trace()
        if self.transform !=None:
            seed=np.random.randint(1,1000)
            random.seed(seed) 
            y=self.transform(image=y)['image']
            random.seed(seed) 
            anothery=self.transform(image=anothery)['image']
            random.seed(seed) 
            sys=self.transform(image=sys)['image']
            random.seed(None)
        num=os.path.dirname(file_name)
        if num in self.train_list:
            return y,anothery,sys,"train"
        else:
            return y,anothery,sys,"val"
        












"""



class MRIImageSet_split_sys_HCP(Dataset):

    #T1/T2图像在和T1图像平级的一个文件夹中，对应的名字是label_file

    def __init__(self,root_dir,crop_sz=256, transform=None,input_type="T1",label_file="wemt1w_on_t2w_masked.nii",index_symbol="SMR",index_MAX=3,train_size=0.8,random_seed=None):
        assert input_type=="T1" or "T2","the type of input image should be T1 or T2"
        self.root_dir = root_dir
        self.file_list=[]
        self.SMRfolder=[]
        for SMRfolder in os.listdir(root_dir):
            self.SMRfolder.append(SMRfolder)
            self.file_list.extend(file for file in os.listdir(os.path.join(root_dir)) if os.path.isdir(file))
        #self.file_list=[file for file in os.listdir(root_dir) if file.endswith(".nii") and input_type in file]
        self.dat_norm_fact = 255 
        self.crop_sz=crop_sz
        self.transform=transform
        self.index_MAX=index_MAX#维度的最大位数
        assert train_size<1 and train_size>0,"train_size should be in the range [0,1]"
        self.random_seed=random_seed
        self.train_size=train_size
        self.index_symbol=index_symbol
        self.train_list,self.val_list=self.train_test_spilt()
        self.label_file=label_file
        print(self.train_list)
        import pdb
       #pdb.set_trace()
        
    def __len__(self):
        num = len(self.file_list)
        return num
    
    def train_test_spilt(self):
        SMR_list=[]
        seed=self.random_seed
        if seed!=None:
            random.seed(seed)
        #get list
        for num in self.SMRfolder:
   
            num_true=num[:6]
            if  num_true not in SMR_list:
                SMR_list.append(num_true)
        length=len(SMR_list)
        print(SMR_list)
        train_length=round(length*self.train_size)
        print(f"{length} groups are divided by {self.index_symbol} ")
        train_list=random.sample(SMR_list,train_length)
        val_list=[elem for elem in SMR_list if elem not in train_list]
        return train_list,val_list
    def __getitem__(self, index):
        file_name=self.file_list[index]
        #sub=file_name[:file_name.find("_")] if self.label==1 else 0
        #y = Image.open(os.path.join(self.root_dir,file_name))
        #pdb.set_trace()
        
        img_path=os.path.join(self.root_dir,file_name)
        img_nib = nib.load(img_path)
        
        y = img_nib.get_fdata() 
        sys_path=os.path.join(img_path.split(".")[0],self.label_file)
        sys_nib=nib.load(sys_path)
      
        sys = sys_nib.get_fdata() 
        #pdb.set_trace()
        if self.transform !=None:
            seed=np.random.randint(1,1000)
            random.seed(seed) 
            y=self.transform(image=y)
            random.seed(seed) 
            sys=self.transform(image=sys)
            random.seed(None)
        num=os.path.dirname(file_name)
        if num in self.train_list:
            return y,sys,"train"
        else:
            return y,sys,"val"""