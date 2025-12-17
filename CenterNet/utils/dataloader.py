import math
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input

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

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class CenternetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(CenternetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)

        self.input_shape        = input_shape
        self.output_shape       = (int(input_shape[0]/4) , int(input_shape[1]/4))
        self.num_classes        = num_classes
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        #-------------------------------------------------#
        #   进行数据增强
        #-------------------------------------------------#
        image, raw, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)

        batch_hm        = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh        = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg       = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask  = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        
        if len(box) != 0:
            boxes = np.array(box[:, :4],dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)

        for i in range(len(box)):
            bbox    = boxes[i].copy()
            cls_id  = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                #-------------------------------------------------#
                #   计算真实框所属的特征点
                #-------------------------------------------------#
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                #----------------------------#
                #   绘制高斯热力图
                #----------------------------#
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                #---------------------------------------------------#
                #   计算宽高真实值
                #---------------------------------------------------#
                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                #---------------------------------------------------#
                #   计算中心偏移量
                #---------------------------------------------------#
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                #---------------------------------------------------#
                #   将对应的mask设置为1
                #---------------------------------------------------#
                batch_reg_mask[ct_int[1], ct_int[0]] = 1

        image = np.transpose(preprocess_input(image), (2, 0, 1))
        raw   = np.transpose(np.array(raw/65535.0,dtype = np.float32), (2, 0, 1))
        return image,raw, batch_hm, batch_wh, batch_reg, batch_reg_mask


    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)

        raw_path=convert_path(line[0])   ###转换路径名
        raw = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
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
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            resized_raw = cv2.resize(raw, (nw,nh), interpolation=cv2.INTER_LINEAR)
            new_raw = np.ones((h, w, 3), dtype=np.float32) * 32768
            new_raw[ dy:dy+nh , dx:dx+nw ] = resized_raw
            raw_data=np.array(new_raw , dtype=np.float32)
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

            return image_data, raw_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)

        image = image.resize((nw,nh), Image.BICUBIC)
        resized_raw = cv2.resize(raw, (nw,nh), interpolation=cv2.INTER_LINEAR)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))


        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        if dx<0:
            resized_raw_start_x=abs(dx)
        else:
            resized_raw_start_x=0

        if dy<0:    
            resized_raw_start_y=abs(dy)
        else:
            resized_raw_start_y=0   

        # 计算粘贴的结束位置，确保不会超出目标图像的边界
        end_x = min(dx + nw, w)  # 源图像的底部不会超过目标图像的右边界
        end_y = min(dy + nh, h)  # 源图像的右边不会超过目标图像的下边界
        # 计算实际粘贴的区域
        start_x = max(dx, 0)  # 源图像的顶部不会超过目标图像的上边界
        start_y = max(dy, 0)  # 源图像的左边不会超过目标图像的左边界

        resized_raw_end_x = end_x - start_x +resized_raw_start_x # 实际粘贴的宽度
        resized_raw_end_y = end_y - start_y +resized_raw_start_y # 实际粘贴的高度


        new_raw = np.ones((h, w, 3), dtype=np.float32) * 32768
        new_raw[start_y:end_y, start_x:end_x, :] = resized_raw[resized_raw_start_y:resized_raw_end_y, resized_raw_start_x:resized_raw_end_x, :]
        raw = new_raw
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            raw = raw[:, ::-1]
        image_data      = np.array(image, np.float32)
        raw_data        = np.array(raw , dtype=np.float32)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        # r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # #---------------------------------#
        # #   将图像转到HSV上
        # #---------------------------------#
        # hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        # dtype           = image_data.dtype
        # #---------------------------------#
        # #   应用变换
        # #---------------------------------#
        # x       = np.arange(0, 256, dtype=r.dtype)
        # lut_hue = ((x * r[0]) % 180).astype(dtype)
        # lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        # lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

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
        
        return image_data,raw_data, box

# DataLoader中collate_fn使用
def centernet_dataset_collate(batch):
    imgs, raws, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], [], []

    for img, raw, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        raws.append(raw)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs            = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    raws            = torch.from_numpy(np.array(raws)).type(torch.FloatTensor)
    batch_hms       = torch.from_numpy(np.array(batch_hms)).type(torch.FloatTensor)
    batch_whs       = torch.from_numpy(np.array(batch_whs)).type(torch.FloatTensor)
    batch_regs      = torch.from_numpy(np.array(batch_regs)).type(torch.FloatTensor)
    batch_reg_masks = torch.from_numpy(np.array(batch_reg_masks)).type(torch.FloatTensor)
    return imgs,raws, batch_hms, batch_whs, batch_regs, batch_reg_masks

