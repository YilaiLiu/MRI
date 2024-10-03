import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self,inch=1,outch=None,act=nn.LeakyReLU(),norm=nn.InstanceNorm3d,after=False):
        super(ConvBlock,self).__init__()
        if outch==None:
            outch=2*inch
        self.act=act
        self.conv=nn.Conv3d(inch,outch,kernel_size=4,stride=2,padding=1)
        self.norm=norm(outch)
        self.after=after
    def forward(self,x):
        x1=self.act(x)
        x2=self.conv(x1)
        y=self.norm(x2)
        if self.after:
            return y,x2  
        else:
            return y,x1

class ConvBlockwithEmb(nn.Module):
    def __init__(self,inch=1,outch=None,act=nn.LeakyReLU(),norm=nn.InstanceNorm3d,after=False):
        super(ConvBlockwithEmb,self).__init__()
        if outch==None:
            outch=2*inch
        self.act=act
        self.conv=nn.Conv3d(inch,outch,kernel_size=4,stride=2,padding=1)
        self.norm=norm
        self.after=after
    def forward(self,x,style_feat):
        x1=self.act(x)
        x2=self.conv(x1)
        #import pdb;pdb.set_trace()
        y=self.norm(x2,style_feat)
        if self.after:
            return y,x2  
        else:
            return y,x1

class EncoderBackbone(nn.Module):
    def __init__(self,inch=1,topnum=64,act=nn.LeakyReLU()):
        super(EncoderBackbone,self).__init__()
        self.inConv=nn.Conv3d(inch,topnum,kernel_size=3,stride=1,padding=1)
        self.conv1=ConvBlock(topnum,act=act)
        self.conv2=ConvBlock(2*topnum,act=act)
        self.conv3=ConvBlock(4*topnum,act=act,after=True)
        self.conv4=ConvBlock(8*topnum,8*topnum,norm=nn.Identity,act=act,after=True)
    def forward(self,x):
        
        x=self.inConv(x)
        x,res1=self.conv1(x)
        x,res2=self.conv2(x)
        x,res3=self.conv3(x)
        x,res4=self.conv4(x)
        #import pdb;pdb.set_trace()
        return x,[res1,res2,res3,res4]

class First_DeConvBlock(nn.Module):
    def __init__(self,inch,outch=None,act=nn.ReLU,norm=nn.InstanceNorm3d):
        super(First_DeConvBlock,self).__init__()
        if outch==None:
            outch=inch
        self.act=act()
        self.conv=nn.ConvTranspose3d(inch,outch,kernel_size=3,stride=1,padding=1)
        self.norm=norm  
    def forward(self,x,res):
    
        x=self.conv(self.act(x))
        #import pdb;pdb.set_trace()
        if res!=None:
            x=torch.cat([x,res],dim=1)
        x=self.norm(x)
        return x
class DeConvBlock(nn.Module):
    def __init__(self,inch=1,outch=None,act=nn.ReLU,norm=nn.InstanceNorm3d,dropout=True):
        super(DeConvBlock,self).__init__()
        if outch==None:
            outch=inch
        self.act=act()
        self.conv=nn.ConvTranspose3d(inch,outch,kernel_size=4,stride=2,padding=1)
        self.norm=norm(outch)
        self.drop=nn.Dropout3d() if dropout else nn.Identity()
    def forward(self,x,res):
        #import pdb;pdb.set_trace()
        x=self.act(x)
        x=self.norm(self.conv(x))
        if res!=None:
            x=torch.cat([x,res],dim=1)
        y=self.drop(x)
        return y

class DeConvBlock_double(nn.Module):
    def __init__(self,inch=1,outch=None,act=nn.ReLU,norm=nn.InstanceNorm3d,dropout=True):
        super(DeConvBlock_double,self).__init__()
        if outch==None:
            outch=inch
        self.act=act()
        self.conv=nn.ConvTranspose3d(inch,inch//2,kernel_size=4,stride=2,padding=1)
        self.norm=norm(outch)
        self.act2=act()
        self.conv2=nn.ConvTranspose3d(inch//2,outch,kernel_size=4,stride=2,padding=1)
        self.norm2=norm(outch)

        self.drop=nn.Dropout3d() if dropout else nn.Identity()
    def forward(self,x,res):
        #import pdb;pdb.set_trace()
        x=self.act(x)
        x=self.norm(self.conv(x))
        x=self.act2(x)
        x=self.norm2(self.conv2(x))
        if res!=None:
            x=torch.cat([x,res],dim=1)
        y=self.drop(x)
        return y
    
class finalConv(nn.Module):
    def __init__(self,inch,outch=1,act=nn.ReLU,finalact=nn.Tanh):
        super(finalConv,self).__init__()
        self.act=act
        self.conv=nn.ConvTranspose3d(inch,outch,kernel_size=1)
        self.finalact=finalact()
    def forward(self,x):
        x=self.act(x)
        y=self.finalact(self.conv(x))
        return y

class DecoderBackbone(nn.Module):
    def __init__(self,inch=512,outch=1,act=nn.LeakyReLU):
        super(DecoderBackbone,self).__init__()
       
        self.conv1=First_DeConvBlock(inch)
        self.conv2=DeConvBlock(2*inch,inch)
        self.conv3=DeConvBlock_double(2*inch,inch//4)
        self.conv4=DeConvBlock(inch//2,inch//8,norm=nn.Identity)
        self.finalconv=finalConv(inch//4,1,act=act)
    def forward(self,x,res):
        x=self.conv1(x,res[3])
        x=self.conv2(x,res[2])
        x=self.conv3(x,res[1])
        x=self.conv4(x,res[0])
        x=self.finalconv(x)
        return x



class Unet_convert(nn.Module):
    def __init__(self,inch=1,outch=1,act=nn.LeakyReLU(negative_slope=0.2)):
        super(Unet_convert,self).__init__()
        self.encoder=EncoderBackbone(inch,act=act)
        self.decoder=DecoderBackbone(outch=outch,act=act)
    def forward(self,x):
        x,res=self.encoder(x)
        x=self.decoder(x,res)
        return x

class Embedding_Block(nn.Module):
    def __init__(self,in_features=128,out_features=128,act=nn.LeakyReLU(negative_slope=0.2)):
        super(Embedding_Block,self).__init__()
        self.Linear1=nn.Linear(in_features,out_features)
        self.Linear2=nn.Sequential(nn.ReLU(),nn.Linear(in_features,out_features))
        self.act=act
        #import pdb;pdb.set_trace()
    def forward(self,x):
        x1=self.Linear1(x).unsqueeze(-1)
        x2=self.Linear2(x).unsqueeze(-1)
        #import pdb;pdb.set_trace()
        x=torch.cat([x1,x2],dim=-1)
        x=self.act(x)#1*embedding_channel*2(一个是mean一个是var)
        return x

class Embedding_layer_Encoder(nn.Module):
    def __init__(self,in_features=1,mid_features=128,act=nn.LeakyReLU(negative_slope=0.2)):
        super(Embedding_layer_Encoder,self).__init__()
        self.embed1=Embedding_Block(mid_features,128,act=act)
        self.embed2=Embedding_Block(mid_features,256,act=act)
        self.embed3=Embedding_Block(mid_features,512,act=act)   
        self.embed4=Embedding_Block(mid_features,512,act=act)
    def forward(self,x):
        vec1=self.embed1(x)
        vec2=self.embed2(x)
        vec3=self.embed3(x)
        vec4=self.embed4(x)
        
        return [vec1,vec2,vec3,vec4]

class Embedding_layer_Decoder(nn.Module):
    def __init__(self,in_features=1,mid_features=128,act=nn.LeakyReLU(negative_slope=0.2)):
        super(Embedding_layer_Decoder,self).__init__()
        self.embed1=Embedding_Block(mid_features,1024,act=act)
        self.embed2=Embedding_Block(mid_features,1024,act=act)
        self.embed3=Embedding_Block(mid_features,512,act=act)   
        self.embed4=Embedding_Block(mid_features,256,act=act)
        self.embed5=Embedding_Block(mid_features,128,act=act)
    def forward(self,x):
        vec1=self.embed1(x)
        vec2=self.embed2(x)
        vec3=self.embed3(x)
        vec4=self.embed4(x)
        vec5=self.embed5(x)
        
        return [vec1,vec2,vec3,vec4,vec5]

def calc_mean_std(feat, eps=1e-5):
    # eps是一个小值，用于避免方差为零时的除以零错误。
    size = feat.size()
    assert (len(size) == 5)
    N, C = size[:2]
    # 将特征图转换为形状为[N, C, H, W]的张量，并计算每个通道的方差
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # 计算每个通道的标准差，并将形状转换为[N, C, 1, 1]
    feat_std = feat_var.sqrt().view(N, C, 1, 1,1)
    # 计算每个通道的均值，并将形状转换为[N, C, 1, 1]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1,1)
    # 返回特征的均值和标准差
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = torch.chunk(style_feat,2,dim=-1)
    content_mean, content_std = calc_mean_std(content_feat)
    N, C = size[:2]
    style_std = style_std.view(N, C, -1).mean(dim=2).view(N, C, 1, 1,1)
    style_mean = style_mean.view(N, C, -1).mean(dim=2).view(N, C, 1, 1,1)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)





class EncoderBackbone_withEmb(nn.Module):
    def __init__(self,inch=1,topnum=64,act=nn.LeakyReLU(),norm=adaptive_instance_normalization):
        super(EncoderBackbone_withEmb,self).__init__()
        self.inConv=nn.Conv3d(inch,topnum,kernel_size=3,stride=1,padding=1)
        self.conv1=ConvBlockwithEmb(topnum,act=act,norm=norm)
        self.conv2=ConvBlockwithEmb(2*topnum,act=act,norm=norm)
        self.conv3=ConvBlockwithEmb(4*topnum,act=act,after=True,norm=norm)
        self.conv4=ConvBlockwithEmb(8*topnum,8*topnum,norm=norm,act=act,after=True)
    def forward(self,x,style_feat):
        
        x=self.inConv(x)
        x,res1=self.conv1(x,style_feat[0])
        x,res2=self.conv2(x,style_feat[1])
        x,res3=self.conv3(x,style_feat[2])
        x,res4=self.conv4(x,style_feat[3])
        #import pdb;pdb.set_trace()
        return x,[res1,res2,res3,res4]
    
class First_DeConvBlock_withEmb(nn.Module):
    def __init__(self,inch,outch=None,act=nn.ReLU,norm=adaptive_instance_normalization):
        super(First_DeConvBlock_withEmb,self).__init__()
        if outch==None:
            outch=inch
        self.act=act()
        self.conv=nn.ConvTranspose3d(inch,outch,kernel_size=3,stride=1,padding=1)
        self.norm=norm

  
    def forward(self,x,res,style_feat):
    
        x=self.conv(self.act(x))
        #import pdb;pdb.set_trace()
        if res!=None:
            x=torch.cat([x,res],dim=1)
        
        x=self.norm(x,style_feat)
        return x


class DeConvBlock_withEmb(nn.Module):
    def __init__(self,inch=1,outch=None,act=nn.ReLU,norm=adaptive_instance_normalization,dropout=True):
        super(DeConvBlock_withEmb,self).__init__()
        if outch==None:
            outch=inch
        self.act=act()
        self.conv=nn.ConvTranspose3d(inch,outch,kernel_size=4,stride=2,padding=1)
        self.norm=norm
        self.drop=nn.Dropout3d() if dropout else nn.Identity()
    def forward(self,x,res,style_feat):
        #import pdb;pdb.set_trace()
        x=self.act(x)
        x=self.conv(x)
        if res!=None:
            x=torch.cat([x,res],dim=1)
        y=self.drop(self.norm(x,style_feat))
        return y
class DeConvBlock_double_withEmb(nn.Module):
    def __init__(self,inch=1,outch=None,act=nn.ReLU,norm=adaptive_instance_normalization,dropout=True):
        super(DeConvBlock_double_withEmb,self).__init__()
        if outch==None:
            outch=inch
        self.act=act()
        self.conv=nn.ConvTranspose3d(inch,inch//2,kernel_size=4,stride=2,padding=1)
        self.norm=norm
        self.act2=act()
        self.conv2=nn.ConvTranspose3d(inch//2,outch,kernel_size=4,stride=2,padding=1)
        self.norm2=norm

        self.drop=nn.Dropout3d() if dropout else nn.Identity()
    def forward(self,x,res,style_feat1,style_feat2):
        #import pdb;pdb.set_trace()
        x=self.act(x)
        x=self.norm(self.conv(x),style_feat1)
        x=self.act2(x)
        x=self.conv2(x)
        if res!=None:
            x=torch.cat([x,res],dim=1)
  
        y=self.drop(self.norm2(x,style_feat2))
        return y
class DecoderBackbone_withEmb(nn.Module):
    def __init__(self,inch=512,outch=1,act=nn.LeakyReLU,norm=adaptive_instance_normalization):
        super(DecoderBackbone_withEmb,self).__init__()
       
        self.conv1=First_DeConvBlock_withEmb(inch,norm=norm)
        self.conv2=DeConvBlock_withEmb(2*inch,inch,norm=norm)
        self.conv3=DeConvBlock_double_withEmb(2*inch,inch//4,norm=norm)
        self.conv4=DeConvBlock_withEmb(inch//2,inch//8,norm=norm)
        self.finalconv=finalConv(inch//4,1,act=act)
    def forward(self,x,res,style_feats):
        x=self.conv1(x,res[3],style_feats[0])
        x=self.conv2(x,res[2],style_feats[1])
        x=self.conv3(x,res[1],style_feats[2],style_feats[3])
        x=self.conv4(x,res[0],style_feats[4])
        x=self.finalconv(x)
        return x



class Unet_Embedded(nn.Module):
    def __init__(self,inch=1,outch=1,act=nn.LeakyReLU(negative_slope=0.2)):
        super(Unet_Embedded,self).__init__()
        self.encoder=EncoderBackbone_withEmb(inch,act=act,norm=adaptive_instance_normalization)
        self.decoder=DecoderBackbone_withEmb(outch=outch,act=act,norm=adaptive_instance_normalization)
        
        self.embed_encoder_T1=Embedding_layer_Encoder()
        self.embed_encoder_T2=Embedding_layer_Encoder()
        self.embed_decoder_T1=Embedding_layer_Decoder()
        self.embed_decoder_T2=Embedding_layer_Decoder()
        self.embed_decoder_ratio=Embedding_layer_Decoder()
        self.embed_encoders = {
            "T1": self.embed_encoder_T1,
            "T2": self.embed_encoder_T2,
        }
        self.embed_decoders={
            "T1": self.embed_decoder_T1,
            "T2": self.embed_decoder_T2,
            "ratio":self.embed_decoder_ratio
        }
        self.adult_vec=nn.Parameter(torch.randn(1,128))
        self.elder_vec=nn.Parameter(torch.randn(1,128))
        self.neonate_vec=nn.Parameter(torch.randn(1,128))
    def forward(self,x,cls="adult",input="T1",output="T2"):
        if cls=="adult":
            vec=self.adult_vec
        elif cls=="neonate":
            vec=self.neonate_vec
        elif cls=="elder":
            vec=self.elder_vec
        else:
            raise ValueError("cls should have one of the above name: adult/neonate/elder")
        embed_encoder=self.embed_encoders.get(input)
        embed_decoder=self.embed_decoders.get(output)
        encoder_embedding=embed_encoder(vec)
        decoder_embedding=embed_decoder(vec)

        x,res=self.encoder(x,encoder_embedding)
        x=self.decoder(x,res,decoder_embedding)
        return x







""" 
    class Unet(nn.Module):
        def __init__(self,inch,outch,act=nn.LeakyReLU):
            self.encoder=
            self.decoder_convert=
            self.decoder_trans=
        def forward(self,x)"""


if __name__=="__main__":
    x=torch.randn([1,1,80,80,80]).cuda()
    model=Unet_Embedded().cuda()
    
    print(model(x).shape)

