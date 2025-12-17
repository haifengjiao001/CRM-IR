import cv2
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input,preprocess_rgb_input
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

class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape = [600, 600], train = True):
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image,image_rgb, y    = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image_rgb   = np.transpose(preprocess_rgb_input(np.array(image_rgb, dtype=np.float32)), (2, 0, 1))
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data    = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box         = box_data[:, :4]
        label       = box_data[:, -1]
        return image, image_rgb,box, label

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像,原来是PIL，现在是cv2
        #------------------------------#
        # image   = Image.open(line[0])
        # image   = cvtColor(image)
        image_rgb = cv2.imread(line[0], cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        raw_path=convert_path(line[0])   ###转换路径名
        image = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        ih,iw,_  = image.shape
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
            #   将图像多余的部分加上灰条，原来是PIL，现在是cv2
            #---------------------------------#
            # image       = image.resize((nw,nh), Image.BICUBIC)
            # new_image   = Image.new('RGB', (w,h), (128,128,128))
            # new_image.paste(image, (dx, dy))
            # image_data  = np.array(new_image, np.float32)
            image_rgb = cv2.resize(image_rgb, (nw,nh), interpolation=cv2.INTER_LINEAR)
            image_rgb_data = np.ones((h, w, 3), dtype=np.float32) * 128
            image_rgb_data[ dy:dy+nh , dx:dx+nw ] = image_rgb

            image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
            image_data = np.ones((h, w, 3), dtype=np.float32) * 32768
            image_data[ dy:dy+nh , dx:dx+nw ] = image


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

            return image_data, image_rgb_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲，原来是PIL，现在是cv2
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        # image = image.resize((nw,nh), Image.BICUBIC)
        image_rgb = cv2.resize(image_rgb, (nw,nh), interpolation=cv2.INTER_LINEAR)
        image     = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)

        #------------------------------------------#
        #   将图像多余的部分加上灰条，原来是PIL，现在是cv2
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        # new_image = Image.new('RGB', (w,h), (128,128,128))
        # new_image.paste(image, (dx, dy))
        # image = new_image
        if dx<0:
            resized_raw_start_x=abs(dx)
        else:
            resized_raw_start_x=0

        if dy<0:    
            resized_raw_start_y=abs(dy)
        else:
            resized_raw_start_y=0   

        # 计算粘贴的结束位置，确保不会超出目标图像的边界
        end_x = min(dx + nw, w)  # 源图像的底部不会超过目标图像的下边界
        end_y = min(dy + nh, h)  # 源图像的右边不会超过目标图像的右边界
        # 计算实际粘贴的区域
        start_x = max(dx, 0)  # 源图像的顶部不会超过目标图像的上边界
        start_y = max(dy, 0)  # 源图像的左边不会超过目标图像的左边界

        resized_raw_end_x = end_x - start_x +resized_raw_start_x # raw的结束位置
        resized_raw_end_y = end_y - start_y +resized_raw_start_y # raw的结束位置

        new_rgb = np.ones((h, w, 3), dtype=np.float32) * 128
        new_rgb[start_y:end_y, start_x:end_x, :] = image_rgb[resized_raw_start_y:resized_raw_end_y, resized_raw_start_x:resized_raw_end_x, :]
        image_rgb = new_rgb

        new_raw = np.ones((h, w, 3), dtype=np.float32) * 32768
        new_raw[start_y:end_y, start_x:end_x, :] = image[resized_raw_start_y:resized_raw_end_y, resized_raw_start_x:resized_raw_end_x, :]
        image = new_raw

        #------------------------------------------#
        #   翻转图像，原来是PIL，现在是cv2
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            # image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_rgb = image_rgb[:, ::-1]
            image     = image[:, ::-1]

        image_rgb_data = image_rgb    
        image_data = image
        # image_rgb_data      = np.array(image_rgb_data, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        # r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # #---------------------------------#
        # #   将图像转到HSV上
        # #---------------------------------#
        # hue, sat, val   = cv2.split(cv2.cvtColor(image_rgb_data, cv2.COLOR_RGB2HSV))
        # dtype           = image_rgb_data.dtype
        # #---------------------------------#
        # #   应用变换
        # #---------------------------------#
        # x       = np.arange(0, 256, dtype=r.dtype)
        # lut_hue = ((x * r[0]) % 180).astype(dtype)
        # lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        # lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # image_rgb_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # image_rgb_data = cv2.cvtColor(image_rgb_data, cv2.COLOR_HSV2RGB)

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
        
        return image_data,image_rgb_data, box

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    images_rgb =[]
    bboxes = []
    labels = []
    for img,rgb ,box, label in batch:
        images.append(img)
        images_rgb.append(rgb)
        bboxes.append(box)
        labels.append(label)
    images = torch.from_numpy(np.array(images))
    images_rgb = torch.from_numpy(np.array(images_rgb))
    return images,images_rgb, bboxes, labels

