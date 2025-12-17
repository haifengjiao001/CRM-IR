from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.utils import save_image
from nets.darknet import darknet53
import math
from nets.adaptive_module import Adaptive_Module
from nets.no_enhance_module import No_Enhance_Module
from compressai.zoo import image_models
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

        num_pixels = N * H * W

        bpploss = sum(
            (torch.log(likelihoods) / (-math.log(2) * num_pixels)).sum()
            for likelihoods in output["likelihoods"].values()
        )

        return bpploss
    
def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False, kernel_nums=8,kernel_size=3,Gtheta = [0.6, 0.8]):
        super(YoloBody, self).__init__()

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
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, raw, rgb,mode="forward"):
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
        
        
        # x_tm=self.noMGE(rawfearture,rgb)
        x_tm = self.TMM(rawfearture, rgb)
        
        # RAW_path="RAW.png"
        # save_rgb_tensor_as_16bit_png_cv2(raw,RAW_path)
        # RGB_path="RGB.png"
        # save_rgb_tensor_as_16bit_png_cv2(rgb,RGB_path)
        # x_tm_path="x_tm.png"
        # save_rgb_tensor_as_16bit_png_cv2(x_tm,x_tm_path)
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x_tm)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        return (out0, out1, out2),bpploss