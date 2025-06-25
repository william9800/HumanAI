from PIL import Image

# === 输入你的 2 张图片路径 ===
img_paths = [
    r'E:\HumanAI\模型4可视化测试结果（主要是针对模型进行比较）/other_model.png',
    r'E:\HumanAI\模型4可视化测试结果（主要是针对模型进行比较）/AUC.png'
]

# 打开图片并转换为 RGB
imgs = [Image.open(p).convert('RGB').copy() for p in img_paths]

# 打印原始尺寸
for idx, img in enumerate(imgs):
    print(f"原始图片 {idx+1} 尺寸: {img.size}")

# 设置统一高度
target_height = 800
for i in range(len(imgs)):
    if imgs[i].height != target_height:
        new_width = int(imgs[i].width * target_height / imgs[i].height)
        imgs[i] = imgs[i].resize((new_width, target_height))
        print(f"已调整图片 {i+1} 为: {imgs[i].size}")

# 计算总宽度
total_width = sum(img.width for img in imgs)
new_img = Image.new('RGB', (total_width, target_height), color=(255, 255, 255))

# 横向粘贴图片
x_offset = 0
for idx, img in enumerate(imgs):
    new_img.paste(img, (x_offset, 0))
    print(f"✅ 图片 {idx+1} 已粘贴，偏移量到: {x_offset}")
    x_offset += img.width

# 保存
new_img.save('combined_horizontal.png')
print("✅ 横向拼接完成，已保存为 combined_horizontal.png")
