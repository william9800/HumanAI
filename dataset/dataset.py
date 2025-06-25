import os
import random
from PIL import Image
from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import transforms as T

class HumanAI(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        只使用 'real' 和 'fake' 作为类别，递归获取所有图片路径，支持健壮的数据划分
        """
        self.test = test
        self.transforms = transforms

        self.classes = ["real", "fake"]
        self.class_to_idx = {"real": 0, "fake": 1}
        self.imgs = []

        # 遍历 real/fake 文件夹及子目录，收集图片路径和标签
        for cls_name in self.classes:
            cls_folder = os.path.join(root, cls_name)
            if not os.path.exists(cls_folder):
                raise ValueError(f"❌ 缺少类别文件夹: {cls_folder}，请检查数据集结构！")

            for root_dir, _, files in os.walk(cls_folder):
                for fname in files:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root_dir, fname)
                        self.imgs.append((img_path, self.class_to_idx[cls_name]))

        if len(self.imgs) == 0:
            raise ValueError(f"❌ 数据集 {root} 为空！请检查是否存在图片。")

        # 如果是 test 模式，不进行划分
        if self.test:
            return

        # 尝试进行训练/验证划分
        try:
            train_imgs, val_imgs = train_test_split(
                self.imgs,
                test_size=0.2,
                random_state=42,
                stratify=[label for _, label in self.imgs]
            )
            self.imgs = train_imgs if train else val_imgs
        except ValueError as e:
            print(f"⚠️ train_test_split() 失败: {e}")
            print("👉 使用所有数据，不进行划分")
            self.imgs = self.imgs

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 无法加载图像: {img_path}，错误: {e}")
            return None, None
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
