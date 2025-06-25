import os
import random
import shutil

# ================= 配置路径 ===================
source_folder = "F:\毕业设计复合训练数据集/train/real/FFHQ"  # 原始图片文件夹
#F:\毕业设计复合训练数据集/train/fake\



destination_folder = "E:\HumanAI\data/train/real"  # 目标文件夹
num_images = 500  # 需要复制的图片数量
clear_destination = True  # 是否在选取前清空目标文件夹 (True: 清空, False: 保留)

# ================= 清空目标文件夹（可选） ===================
if clear_destination and os.path.exists(destination_folder):
    for file in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, file)
        if os.path.isfile(file_path):  # 仅删除文件，不删除子文件夹
            os.remove(file_path)
    print(f"🗑️ 目标文件夹 {destination_folder} 已清空！")
# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 获取所有图片
all_images = [img for img in os.listdir(source_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 确保图片足够
if len(all_images) < num_images:
    raise ValueError(f"源文件夹中只有 {len(all_images)} 张图片，不足 {num_images} 张！")

# 随机选择 1000 张图片
selected_images = random.sample(all_images, num_images)

# 复制到目标文件夹
for img in selected_images:
    src_path = os.path.join(source_folder, img)
    dest_path = os.path.join(destination_folder, img)
    shutil.copy2(src_path, dest_path)  # 保留元数据

print(f"✅ 已成功随机选择 {num_images} 张图片，并复制到 {destination_folder} 🎉")
