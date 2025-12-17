import os
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 定义颜色生成函数
def generate_colors(num_classes):
    hsv_tuples = [(x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors

# 定义类别
classes = ['car', 'motorbike', 'bicycle', 'chair', 'diningtable', 'bottle', 'tvmonitor', 'bus']
colors = generate_colors(len(classes))

# 定义注释框的信息
# 格式为 [(label, top, left, bottom, right), ...]

# 146_xtm.png
annotations = [
('car',0.89, 126 ,607 ,565, 1179),
('chair', 0.89,252, 48 ,618, 335)
]


# 定义输入图像的尺寸（假设为模型的输入尺寸）
input_shape = (416, 416)  # 示例输入尺寸

# 打开原始图像
image_path = '146_xtm.png'  # 替换为您的图像路径
image = Image.open(image_path)
# 提取中间部分 (512, 341)
left = (image.width - 512) // 2
upper = (image.height - 341) // 2
right = left + 512
lower = upper + 341
cropped_image = image.crop((left, upper, right, lower))


# 调整大小到 (1200, 800)
resized_image = cropped_image.resize((1200, 800), Image.ANTIALIAS)

# 动态调整字体大小和线条粗细
font_size = int(np.floor(3e-2 * resized_image.size[1] + 0.5))
thickness = int(max((resized_image.size[0] + resized_image.size[1]) // np.mean(input_shape), 1))

# 加载自定义字体
font_path = "model_data/simhei.ttf"  # 替换为您的字体文件路径
font = ImageFont.truetype(font=font_path, size=font_size)

# 创建绘图对象
draw = ImageDraw.Draw(resized_image)

# 遍历每个注释框
for label,confidence, top, left, bottom, right in annotations:
    class_id = classes.index(label)
    color = colors[class_id]

    # 绘制边界框
    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=color)

    # 添加注释
    text = f"{label} {confidence:.2f}"
    text_size = draw.textsize(text, font=font)
    draw.rectangle([left, top - text_size[1] - 3, left + text_size[0], top], fill=color)
    draw.text((left, top - text_size[1] - 3), text, fill=(0, 0, 0), font=font)

# 保存结果图像
output_path = 'visual_xtm_146.jpg'  # 替换为您希望保存的路径
resized_image.save(output_path)

print(f"注释完成，已保存到 {output_path}")