import torch
import torch.nn as nn

class No_Enhance_Module(nn.Module):
    def __init__(self, in_channels_16, in_channels_3, out_channels, kernel_size, stride=1, padding=0):
        """
        定义一个模块，将16维张量和3维张量拼接后进行卷积操作

        :param in_channels_16: 16维张量的通道数
        :param in_channels_3: 3维张量的通道数
        :param out_channels: 输出张量的通道数
        :param kernel_size: 卷积核大小
        :param stride: 卷积步长
        :param padding: 卷积填充
        """
        super(No_Enhance_Module, self).__init__()
        # 计算拼接后的总通道数
        total_in_channels = in_channels_16 + in_channels_3
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels=total_in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, tensor16, tensor3):
        """
        前向传播函数

        :param tensor16: 16维张量，形状为 (B, 16, H, W)
        :param tensor3: 3维张量，形状为 (B, 3, H, W)
        :return: 输出张量，形状为 (B, out_channels, H, W)
        """
        # 拼接张量
        concat_tensor = torch.cat((tensor16, tensor3), dim=1)
        # 应用卷积操作
        output_tensor = self.conv(concat_tensor)
        return output_tensor