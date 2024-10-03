import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from volumentations import *
import nibabel as nib
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


class AlignedForMRIDataset(BaseDataset):
    """ImageNet dataset.

    Args:
        root_dir (str): Path to ImageNet root directory.
        crop_sz (int): Size to crop loaded images to (passed to 
            torchtransforms.RandomResizedCrop).
        transform (list, default=None): List of transforms to apply to loaded
            images.
    """
    #T1/T2图像在和T1图像平级的一个文件夹中，对应的名字是label_file

    def __init__(self,opt):
        """root_dir,crop_sz=256, transform=None,input_type="T1w_acpc",index_symbol="SMR",index_MAX=3,train_size=0.8,random_seed=None,ratio_name="T1wDividedByT2w_ribbon.nii.gz" """
        assert opt.input_type=="T1w_acpc" or "T2w_acpc","the type of input image should be T1 or T2"
        self.input_type=opt.input_type
        self.other_type="T1w_acpc" if opt.input_type=="T2w_acpc" else "T2w_acpc" 
        self.root_dir = opt.dataroot
        self.ratio_name=opt.ratio_name
        self.SMRfolder=[]
        for SMRfolder in os.listdir(opt.dataroot):
            self.SMRfolder.append(SMRfolder)
        
        self.dat_norm_fact = 255 
        self.crop_sz=opt.crop_size
        self.transform=get_augmentation((opt.crop_size,opt.crop_size,opt.crop_size))
        
        assert opt.train_size<1 and opt.train_size>0,"train_size should be in the range [0,1]"
        self.random_seed=opt.random_seed
        self.train_size=opt.train_size
     
        train_list,val_list=self.train_test_spilt()
        self.phase=opt.phase
        if self.phase=="train":
            self.file_list=train_list
        else:
            self.file_list=val_list
        
            
        """ import pdb
        pdb.set_trace()"""
        
    def __len__(self):
        num = len(self.SMRfolder)
        return num
    
    def train_test_spilt(self):
        SMR_list=[]
        
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
        #print(f"{length} groups are divided by {self.index_symbol} ")
        seed=self.random_seed
        if seed!=None:
            random.seed(seed)
        train_list=random.sample(SMR_list,train_length)
        val_list=[elem for elem in SMR_list if elem not in train_list]
        return train_list,val_list
    def __getitem__(self, index):
        """import pdb;pdb.set_trace()"""
        file_name=self.file_list[index]
        for file in os.listdir(os.path.join(self.root_dir,file_name,"T1w")):
            if file.endswith(".nii") and self.input_type in file:
                img_name=os.path.join(file_name,"T1w",file) 
        #sub=file_name[:file_name.find("_")] if self.label==1 else 0
        #y = Image.open(os.path.join(self.root_dir,file_name))
        #pdb.set_trace()
        img_path=os.path.join(self.root_dir,img_name)
        img_nib = nib.load(img_path)
        for file in os.listdir(os.path.join(self.root_dir,file_name,"T1w")):
            if file.endswith(".nii") and self.other_type in file:
                anotherimg_name=os.path.join(file_name,"T1w",file) 
        anotherimg_path=os.path.join(self.root_dir,anotherimg_name)
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

        return {'A': y, 'B': sys, 'A_paths': self.SMRfolder[index], 'B_paths': self.SMRfolder[index]}
   
        