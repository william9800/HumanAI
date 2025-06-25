import os
import random
import shutil
import glob
import re

source_folder = "F:\毕业设计复合训练数据集/train/real\FFHQ"  # 原始图片文件夹
#F:\毕业设计复合训练数据集/train/fake\



destination_folder = "E:\compression_image"  # 目标文件夹
num_images = 10  # 需要复制的图片数量
clear_destination = True  # 是否在选取前清空目标文件夹 (True: 清空, False: 保留)

# 1️⃣ **清空目标文件夹（如果启用）**
if clear_destination and os.path.exists(destination_folder):
    for file in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"🗑️ 目标文件夹 {destination_folder} 已清空！")

os.makedirs(destination_folder, exist_ok=True)

# 2️⃣ **递归搜索所有子文件夹的图片**
all_images = []

for root, _, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            all_images.append(os.path.normpath(os.path.join(root, file)))  # 规范路径

# **使用 glob.glob() 进一步检查**
all_images += glob.glob(source_folder + "/**/*.jpg", recursive=True)
all_images += glob.glob(source_folder + "/**/*.png", recursive=True)
all_images += glob.glob(source_folder + "/**/*.jpeg", recursive=True)

print(f"📸 找到 {len(all_images)} 张图片")

if len(all_images) < num_images:
    raise ValueError(f"❌ 只有 {len(all_images)} 张图片，不足 {num_images} 张！")

# 3️⃣ **随机选取 `num_images` 张图片**
selected_images = random.sample(all_images, num_images)

# 4️⃣ **复制图片，并重新命名**
for idx, img_path in enumerate(selected_images):
    # 生成新的文件名（image_0001.jpg, image_0002.jpg, ...）
    new_filename = f"image_{idx+1:04d}" + os.path.splitext(img_path)[1]

    # **移除特殊字符（确保 Windows / Linux / Mac 兼容）**
    new_filename = re.sub(r'[\\/:*?"<>|]', '_', new_filename)

    # 目标路径
    dest_path = os.path.normpath(os.path.join(destination_folder, new_filename))

    # 复制文件
    shutil.copy2(img_path, dest_path)

print(f"✅ 成功复制 {num_images} 张图片到 {destination_folder} 🎉")
