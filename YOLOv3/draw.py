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

# 定义文件夹路径
detection_results_folder = 'map_out/detection-results'
visual_picture_folder = 'map_out/visual_picture'
last_results_folder = 'map_out/last_results'

# 确保输出文件夹存在
os.makedirs(last_results_folder, exist_ok=True)

# 遍历 /detection-results 文件夹中的所有 .txt 文件
for filename in os.listdir(detection_results_folder):
    if filename.endswith('.txt'):
        # 读取 .txt 文件
        file_path = os.path.join(detection_results_folder, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 获取对应的图像文件名
        image_filename = f"visual_{os.path.splitext(filename)[0]}.png"
        image_path = os.path.join(visual_picture_folder, image_filename)

        # 打开图像
        image = Image.open(image_path)
        # 提取中间部分 (512, 341)
        left = (image.width - 512) // 2
        upper = (image.height - 341) // 2
        right = left + 512
        lower = upper + 341
        cropped_image = image.crop((left, upper, right, lower))

        # 调整大小到 (1200, 800)
        resized_image = cropped_image.resize((1200, 800), Image.ANTIALIAS)
        image=resized_image
        draw = ImageDraw.Draw(image)

        # 动态调整字体大小和线条粗细
        font_size = int(np.floor(3e-2 * image.size[1] + 0.5))
        thickness = int(max((image.size[0] + image.size[1]) // 416, 1))

        # 加载自定义字体
        font_path = "model_data/simhei.ttf"  # 替换为您的字体文件路径
        font = ImageFont.truetype(font=font_path, size=font_size)

        # 遍历每一行注释信息
        for line in lines:
            parts = line.strip().split()
            label = parts[0]
            confidence = float(parts[1])
            left,top,right,bottom = map(int, parts[2:])
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
        output_filename = f"OD_results_{os.path.splitext(filename)[0]}.png"
        output_path = os.path.join(last_results_folder, output_filename)
        image.save(output_path)

print(f"所有图像已处理并保存到 {last_results_folder}")