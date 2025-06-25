import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
from glob import glob

def preprocess_image(img_path, target_size=(256, 256)):
    """
    预处理图片：调整大小并保持宽高比（填充黑边）

    参数:
        img_path: 图片路径
        target_size: 目标尺寸 (宽, 高)

    返回:
        预处理后的PIL图像对象
    """
    try:
        img = Image.open(img_path)

        # 保持宽高比的缩略图（使用LANCZOS重采样）
        img.thumbnail((target_size[0], target_size[1]), Image.Resampling.LANCZOS)

        # 计算填充位置（居中）
        delta_w = target_size[0] - img.size[0]
        delta_h = target_size[1] - img.size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))

        # 使用ImageOps.expand进行填充（更高效的方法）
        new_img = ImageOps.expand(img, padding, fill='black')
        return new_img
    except Exception as e:
        print(f"预处理图片 {img_path} 时出错: {e}")
        return Image.new('RGB', target_size, (0, 0, 0))

def display_training_images(folder_path, output_file='training_images.png', target_size=(256, 256)):
    """
    自动获取文件夹中的图片，预处理后生成展示表格

    参数:
        folder_path: 图片文件夹路径 (如 'C:/Users/86134/Desktop/图片/')
        output_file: 输出文件名
        target_size: 统一的目标图片尺寸 (宽, 高)
    """
    # 获取所有子文件夹（每个子文件夹代表一个数据集）
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    subfolders.sort()  # 按名称排序

    # 准备数据
    dataset_names = []
    preprocessed_images = []

    for folder in subfolders:
        # 获取数据集名称（使用文件夹名）
        dataset_name = os.path.basename(folder)
        dataset_names.append(dataset_name)

        # 获取该文件夹中的图片（支持多种格式）
        img_files = glob(os.path.join(folder, '*.jpg')) + \
                    glob(os.path.join(folder, '*.png')) + \
                    glob(os.path.join(folder, '*.jpeg')) + \
                    glob(os.path.join(folder, '*.bmp'))
        img_files.sort()  # 按文件名排序
        img_files = img_files[:8]  # 只取前8张

        # 预处理图片
        folder_images = []
        for img_path in img_files:
            if img_path:
                folder_images.append(preprocess_image(img_path, target_size))
            else:
                folder_images.append(Image.new('RGB', target_size, (0, 0, 0)))

        # 如果不足8张，用空白图片补全
        while len(folder_images) < 4:
            folder_images.append(Image.new('RGB', target_size, (0, 0, 0)))

        preprocessed_images.append(folder_images)

    # 创建图形
    rows = len(dataset_names)
    cols = 5  # 1列名称 + 8列图片

    plt.figure(figsize=(20, 3 * rows))
    gs = gridspec.GridSpec(rows, cols, width_ratios=[1]+[2]*4)

    for i in range(rows):
        # 添加数据集名称
        ax = plt.subplot(gs[i, 0])
        ax.text(0.5, 0.5, dataset_names[i],
                ha='center', va='center', fontsize=24)
        ax.axis('off')

        # 添加图片
        for j in range(4   ):
            ax = plt.subplot(gs[i, j+1])
            ax.imshow(preprocessed_images[i][j])
            ax.axis('off')


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"预处理后的图片表格已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际图片文件夹路径
    folder_path = r'C:\Users\86134\Desktop\图片'

    # 设置统一的目标分辨率（宽, 高）
    target_size = (256, 256)  # 可根据需要调整

    display_training_images(folder_path,
                            'newreal.png',
                            target_size)