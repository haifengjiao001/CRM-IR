import math

import torch.nn as nn
import torch

from nets.hourglass import *
from nets.resnet50 import resnet50, resnet50_Decoder, resnet50_Head
from .adaptive_module import Adaptive_Module
from nets.no_enhance_module import No_Enhance_Module
from compressai.zoo import image_models
from torchvision.utils import save_image
from utils.saveimage import save_rgb_tensor_as_16bit_png_cv2
def iter_str_length(list_x):
    string_length = []
    for x in list_x:
        if isinstance(x, (str, bytes)):
            string_length.append(len(x))
        else:
            string_length.extend(iter_str_length(x))
    return string_length

class BppLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, beta=1):
        super().__init__()
        self.lmbda = lmbda
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        
    def forward(self, output, target):
        N, C, H, W = target.size()
        
        # print("N,H,W==",N,H,W)
        num_pixels = N * H * W
        if num_pixels == 0:
            raise ValueError("num_pixels cannot be zero")
        
        for likelihoods in output["likelihoods"].values():
            if torch.any(likelihoods <= 0):
                print("likelihoods contains zero or negative values")
                raise ValueError("likelihoods contains zero or negative values")
        for likelihoods in output["likelihoods"].values():
            if torch.isnan(likelihoods).any() or torch.isinf(likelihoods).any():
                print("likelihoods contains NaN or inf")    
                raise ValueError("llikelihoods contains NaN or inf")
            
        # epsilon = 1e-8    
        bpploss = sum(
            (torch.log(likelihoods) / (-math.log(2) * num_pixels)).sum()
            for likelihoods in output["likelihoods"].values()
        )
        # z = output["z"]
        # z_hat = output["z_hat"]

        # bpploss = bpploss 
        return bpploss

class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes = 20, pretrained = False):
        super(CenterNet_Resnet50, self).__init__()
        self.pretrained = pretrained

        self.compress_model = image_models["learned_context_journal4"](
            quality=1, stride=2, demosaic=False,
            noquant=False, reduce_c=4, hyper_mean=False, 
            resid=False, relu=False, context_z=False, 
            down_num=2, channel_mean=False, resid_path=False,
            sampling_num=4, use_deconv=True, ablation_no_raw=False, 
            z_scale=1, rounding="noise", adaptive_quant=False, 
            lambda_list=None, norm=None, act="GDN", gmm_num=None, 
            drop_pixel=False, soft_gumbel=False, rounding_aux="forward")
        
        self.TMM = Adaptive_Module(in_ch=3, nf=16, gamma_range=[7.0, 10.5])
        # self.noMGE=No_Enhance_Module(in_channels_16=16, in_channels_3=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # 512,512,3 -> 16,16,2048
        self.backbone = resnet50(pretrained = pretrained)
        # 16,16,2048 -> 128,128,64
        self.decoder = resnet50_Decoder(2048)
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        self.head = resnet50_Head(channel=64, num_classes=num_classes)
        
        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)
        
    def forward(self, rgb, raw, mode="forward"):
        # rgb_path="rgb.png"
        # save_image(rgb, rgb_path)
        # raw_path="raw.png"
        # save_image(raw, raw_path)
        # assert 0
        if mode=="forward":
            compress_out=self.compress_model(raw,rgb)
            compress_criterion = BppLoss()
            bpploss = compress_criterion(compress_out, raw)
            rawfearture=compress_out["x_hat"]
        if mode=="test":
            self.compress_model.update()
            compressed_result=self.compress_model.compress(raw,rgb)
            Bytes = sum(iter_str_length(compressed_result["strings"]))
            kb = Bytes/1024
            # print("Real size of compressed string %.5e KB"%kb)
            real_bpp = (8*Bytes/(raw.shape[2]*raw.shape[3]))
            # print("Real bpp: %.5e"%real_bpp)
            rec_from_strings = self.compress_model.decompress(compressed_result["strings"], compressed_result["shape"], rgb, compressed_result = compressed_result)
            rawfearture=rec_from_strings["x_hat"].float()
            bpploss=real_bpp


        # rawfearture_path="rawfearture.png"
        # save_image(rawfearture[0][0], rawfearture_path)
        # print("rawfearture=========",rawfearture)
        # x_tm=self.noMGE(rawfearture,rgb)
        x_tm = self.TMM(rawfearture, rgb)

        x_tm_path="x_tm.png"
        save_rgb_tensor_as_16bit_png_cv2(x_tm,x_tm_path)
        feat = self.backbone(x_tm)
        return self.head(self.decoder(feat))+(bpploss,)

class CenterNet_HourglassNet(nn.Module):
    def __init__(self, heads, pretrained=False, num_stacks=2, n=5, cnv_dim=256, dims=[256, 256, 384, 384, 384, 512], modules = [2, 2, 2, 2, 2, 4]):
        super(CenterNet_HourglassNet, self).__init__()
        if pretrained:
            raise ValueError("HourglassNet has no pretrained model")

        self.nstack    = num_stacks
        self.heads     = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(
                    conv2d(7, 3, 128, stride=2),
                    residual(3, 128, 256, stride=2)
                ) 
        
        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules
            ) for _ in range(num_stacks)
        ])

        self.cnvs = nn.ModuleList([
            conv2d(3, curr_dim, cnv_dim) for _ in range(num_stacks)
        ])

        self.inters = nn.ModuleList([
            residual(3, curr_dim, curr_dim) for _ in range(num_stacks - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])
        
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])

        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    nn.Sequential(
                        conv2d(3, cnv_dim, curr_dim, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], (1, 1))
                    )  for _ in range(num_stacks)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].weight.data.fill_(0)
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    nn.Sequential(
                        conv2d(3, cnv_dim, curr_dim, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], (1, 1))
                    )  for _ in range(num_stacks)
                ])
                self.__setattr__(head, module)


        self.relu = nn.ReLU(inplace=True)

    def freeze_backbone(self):
        freeze_list = [self.pre, self.kps]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        freeze_list = [self.pre, self.kps]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, image):
        # print('image shape', image.shape)
        inter = self.pre(image)
        outs  = []

        for ind in range(self.nstack):
            kp  = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp)

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

            out = {}
            for head in self.heads:
                out[head] = self.__getattr__(head)[ind](cnv)
            outs.append(out)
        return outs
