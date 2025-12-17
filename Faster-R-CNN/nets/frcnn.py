import torch.nn as nn
import math
import torch
from nets.classifier import Resnet50RoIHead, VGG16RoIHead
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16
from nets.adaptive_module import Adaptive_Module
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
        return bpploss
class FasterRCNN(nn.Module):
    def __init__(self,  num_classes,  
                    mode = "training",
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg',
                    pretrained = False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride

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
        #---------------------------------#
        #   一共存在两个主干
        #   vgg和resnet50
        #---------------------------------#
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            #---------------------------------#
            #   构建建议框网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建分类器网络
            #---------------------------------#
            self.head = VGG16RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 7,
                spatial_scale   = 1,
                classifier      = classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.head = Resnet50RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier
            )
            
    def forward(self, x,rgb, scale=1., mode="forward",test="False"):
        # rgb_path="rgb.png"
        # save_rgb_tensor_as_16bit_png_cv2(rgb,rgb_path)
        # raw_path="raw.png"
        # save_rgb_tensor_as_16bit_png_cv2(x,raw_path)

        if mode == "forward":
            #---------------------------------#
            #   计算输入图片的大小
            #---------------------------------#
            img_size        = x.shape[2:]

            if test=="False":
                compress_out=self.compress_model(x,rgb)
                compress_criterion = BppLoss()
                bpploss = compress_criterion(compress_out, x)
                rawfearture=compress_out["x_hat"]
                # print("shape11=====",rawfearture.shape)
                # print("dayin",bpploss.item())
            if test=="True":
                self.compress_model.update()
                compressed_result=self.compress_model.compress(x,rgb)
                Bytes = sum(iter_str_length(compressed_result["strings"]))
                kb = Bytes/1024
                # print("Real size of compressed string %.5e KB"%kb)
                real_bpp = (8*Bytes/(x.shape[2]*x.shape[3]))
                # print("Real bpp: %.5e"%real_bpp)

                rec_from_strings = self.compress_model.decompress(compressed_result["strings"], compressed_result["shape"], rgb, compressed_result = compressed_result)
                rawfearture=rec_from_strings["x_hat"].float()
                bpploss=real_bpp

            # x_tm=self.noMGE(rawfearture,rgb)
            x_tm = self.TMM(rawfearture, rgb)
            # x_tm_path="x_tm.png"
            # save_rgb_tensor_as_16bit_png_cv2(x_tm,x_tm_path)

            base_feature    = self.extractor.forward(x_tm)
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            # base_feature    = self.extractor.forward(x_tm)
            # base_feature    = self.extractor.forward(x)

            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            _, _, rois, roi_indices, _  = self.rpn.forward(base_feature, img_size, scale)
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices ,bpploss
        elif mode == "extractor":
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            
            compress_out=self.compress_model(x,rgb)
            compress_criterion = BppLoss()
            bpploss = compress_criterion(compress_out, x)
            rawfearture=compress_out["x_hat"]
            # print("shape11=====",rawfearture.shape)
            # print("dayin",bpploss.item())
            
            # x_tm=self.noMGE(rawfearture,rgb)
            x_tm = self.TMM(rawfearture, rgb)

            base_feature    = self.extractor.forward(x_tm)
            # base_feature    = self.extractor.forward(x)

            return base_feature ,bpploss
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
