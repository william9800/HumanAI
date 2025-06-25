import os
import random
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#全局参数
Noise_level=30 #0-100
Radius=1.5    #0-10
Block_size=500   #5-Min（W,H）
Factor=1.5  #0.1-2.0
Quality=10  #5-95

def add_noise(img, noise_level=Noise_level):
    """添加高斯噪声"""
    np_img = np.array(img).astype(np.int16)
    noise = np.random.normal(0, noise_level, np_img.shape)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_blur(img, radius=Radius):
    """高斯模糊"""
    return img.filter(ImageFilter.GaussianBlur(radius))

def apply_occlusion(img, block_size=Block_size):
    """随机遮挡图像区域"""
    img_np = np.array(img)
    h, w, _ = img_np.shape
    x = random.randint(0, w - block_size)
    y = random.randint(0, h - block_size)
    img_np[y:y+block_size, x:x+block_size] = 0
    return Image.fromarray(img_np)

def apply_brightness(img, factor=Factor):
    """改变亮度"""
    ycbcr = img.convert('YCbCr')
    y, cb, cr = ycbcr.split()
    y = y.point(lambda i: min(int(i * factor), 255))
    return Image.merge('YCbCr', (y, cb, cr)).convert('RGB')

def jpeg_compress(img, quality=Quality):
    """JPEG 压缩攻击"""
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def apply_cleaning_attack(img, types):
    """组合攻击"""
    if "blur" in types:
        img = apply_blur(img)
    if "noise" in types:
        img = add_noise(img)
    if "occlusion" in types:
        img = apply_occlusion(img)
    if "brightness" in types:
        img = apply_brightness(img, factor=random.uniform(0.3, 0.7))
    if "jpeg" in types:
        img = jpeg_compress(img)
    return img

def attack_folder(input_folder, output_folder, attack_types):
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    for root, _, files in os.walk(input_folder):
        for fname in tqdm(files, desc="🔁 执行干扰攻击"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            src_path = os.path.join(root, fname)
            try:
                img = Image.open(src_path).convert("RGB")
                attacked = apply_cleaning_attack(img, attack_types)

                rel_path = os.path.relpath(src_path, input_folder)
                save_path = os.path.join(output_folder, rel_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                attacked.save(save_path)
                count += 1
            except Exception as e:
                print(f"❌ 跳过 {src_path}，原因: {e}")
    print(f"\n🎯 干扰完成，共生成 {count} 张图片")

# 示例调用
if __name__ == '__main__':
    attack_folder(
        input_folder="E:/compression_image",
        output_folder="E:/compression_image",
        attack_types=["brightness"]  # 可选项: blur, noise, brightness, jpeg, occlusion

    )
