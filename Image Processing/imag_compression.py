import os
from PIL import Image

def compress_images(input_folder, quality=75, max_size=None):
    """
    批量压缩图片（同文件夹压缩，强制保存为 JPEG，并删除非 JPEG 原始图）
    """
    count = 0
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                filepath = os.path.join(root, filename)
                try:
                    img = Image.open(filepath)
                    if max_size:
                        img.thumbnail(max_size, Image.LANCZOS)

                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # 生成保存路径（强制 .jpg 后缀）
                    save_path = os.path.splitext(filepath)[0] + ".jpg"

                    img.save(save_path, format="JPEG", quality=quality)

                    count += 1
                    print(f"✅ 压缩完成: {save_path}")

                    # 如果原始文件不是 save_path（即不同名文件，如 .png → .jpg），删除原文件
                    if os.path.abspath(save_path) != os.path.abspath(filepath):
                        os.remove(filepath)
                        print(f"🗑️ 已删除原始文件: {filepath}")

                except Exception as e:
                    print(f"⚠️ 跳过损坏图片: {filepath}，错误: {e}")

    print(f"\n🎉 共压缩 {count} 张图片")

# 示例
if __name__ == '__main__':
    compress_images(
        input_folder="E:/compression_image",
        quality=20,
        max_size=(800, 800)
    )
