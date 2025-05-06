import os
import shutil

# 输入路径
val_dir = '/root/autodl-tmp/imagenet/val'
imagenet_classes_file = '/root/ConCLIP/Negbench/negbench/benchmarks/src/evaluation/imagenet_classes.txt'

# 读取imagenet_classes.txt中的类别信息
with open(imagenet_classes_file, 'r') as f:
    class_names = [line.strip() for line in f]

# 遍历所有图片并将它们按照类别整理到子文件夹中
for idx, class_name in enumerate(class_names):
    class_dir = os.path.join(val_dir, f'n{str(idx).zfill(8)}')
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # 找到该类的所有图片，并将它们移动到对应的文件夹中
    for img_file in os.listdir(val_dir):
        if img_file.endswith('.JPEG') or img_file.endswith('.jpg') or img_file.endswith('.png'):
            # 这里可以根据图片的命名规则来判断类别，假设文件名中包含类别ID
            # 例如：n01440764_1000.JPEG 表示类别为 n01440764
            if str(idx).zfill(8) in img_file:
                # 移动图片到对应的文件夹
                shutil.move(os.path.join(val_dir, img_file), os.path.join(class_dir, img_file))

print("图片整理完成")
