import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
from utils.utils import cvtColor, preprocess_rgb_input, preprocess_raw_input
def convert_path(input_path):
    # 拆分路径为部分
    parts = input_path.split(os.sep)
    
    # 检查是否以 'JPEGImages' 结尾
    if parts[-2] == 'JPEGImages':
        parts[-2] = 'RAWImages'  # 更改目录名为 'RAWImages'
        
    # 检查文件扩展名是否为 '.jpg'
    if parts[-1].endswith('.jpg'):
        parts[-1] = parts[-1].replace('.jpg', '.png')  # 更改扩展名为 '.png'
    
    # 重新组合路径
    new_path = os.sep.join(parts)
    return new_path

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.length             = len(self.annotation_lines)
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image_raw, image_rgb, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image_raw       = np.transpose(preprocess_raw_input(np.array(image_raw, dtype=np.float32)), (2, 0, 1))
        image_rgb       = np.transpose(preprocess_rgb_input(np.array(image_rgb, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image_raw, image_rgb, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image_rgb = cv2.imread(line[0], cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        raw_path=convert_path(line[0])   ###转换路径名
        image_raw = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        ih,iw,_  = image_rgb.shape
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image_rgb = cv2.resize(image_rgb, (nw,nh), interpolation=cv2.INTER_LINEAR)
            image_rgb_data = np.ones((h, w, 3), dtype=np.float32) * 128
            image_rgb_data[ dy:dy+nh , dx:dx+nw ] = image_rgb

            image_raw = cv2.resize(image_raw, (nw,nh), interpolation=cv2.INTER_LINEAR)
            image_raw_data = np.ones((h, w, 3), dtype=np.float32) * 32768
            image_raw_data[ dy:dy+nh , dx:dx+nw ] = image_raw

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_raw_data,image_rgb_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image_rgb = cv2.resize(image_rgb, (nw,nh), interpolation=cv2.INTER_LINEAR)
        image_raw = cv2.resize(image_raw, (nw,nh), interpolation=cv2.INTER_LINEAR)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        if dx<0:
            resized_start_x=abs(dx)
        else:
            resized_start_x=0

        if dy<0:    
            resized_start_y=abs(dy)
        else:
            resized_start_y=0  

        # 计算粘贴的结束位置，确保不会超出目标图像的边界
        end_x = min(dx + nw, w)  # 源图像的底部不会超过目标图像的下边界
        end_y = min(dy + nh, h)  # 源图像的右边不会超过目标图像的右边界
        # 计算实际粘贴的区域
        start_x = max(dx, 0)  # 源图像的顶部不会超过目标图像的上边界
        start_y = max(dy, 0)  # 源图像的左边不会超过目标图像的左边界

        resized_end_x = end_x - start_x +resized_start_x # 源图像的结束位置
        resized_end_y = end_y - start_y +resized_start_y # 源图像的结束位置

        new_rgb = np.ones((h, w, 3), dtype=np.float32) * 128
        new_rgb[start_y:end_y, start_x:end_x, :] = image_rgb[resized_start_y:resized_end_y, resized_start_x:resized_end_x, :]
        image_rgb = new_rgb

        new_raw = np.ones((h, w, 3), dtype=np.float32) * 32768
        new_raw[start_y:end_y, start_x:end_x, :] = image_raw[resized_start_y:resized_end_y, resized_start_x:resized_end_x, :]
        image_raw = new_raw

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image_rgb = image_rgb[:, ::-1]
            image_raw = image_raw[:, ::-1]

        image_raw_data = image_raw
        image_rgb_data = image_rgb
        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_raw_data,image_rgb_data, box
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images_raw = []
    images_rgb = []
    bboxes = []
    for raw,rgb, box in batch:
        images_raw.append(raw)
        images_rgb.append(rgb)
        bboxes.append(box)
    images_raw = torch.from_numpy(np.array(images_raw)).type(torch.FloatTensor)
    images_rgb = torch.from_numpy(np.array(images_rgb)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images_raw, images_rgb, bboxes
