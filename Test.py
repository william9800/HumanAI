import torch
import torchvision.transforms as transforms
from dataset.dataset import HumanAI
from efficientnet_pytorch import EfficientNet

# 设备选择
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 预处理
transform_test = transforms.Compose([
    transforms.CenterCrop(600),  # EfficientNet-B7推荐尺寸
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 重新构造模型并加载参数
model = EfficientNet.from_pretrained('efficientnet-b7')
num_ftrs = model._fc.in_features
model._fc = torch.nn.Linear(num_ftrs, 2)  # 适配二分类
model.load_state_dict(torch.load("model_final.pth"))  # 加载权重
model.to(DEVICE)
model.eval()

# 预测类别名称
classes = ( "real", "fake")

# 加载测试集
dataset_test = HumanAI('data/test/', transforms=transform_test, test=True)

if len(dataset_test) == 0:
    raise ValueError("❌ 数据集为空，请检查 `data/test/` 是否有图片！")

print(f"📌 测试集中共有 {len(dataset_test)} 张图片")

# 预测
for index, (img, label) in enumerate(dataset_test):
    img = img.unsqueeze(0).to(DEVICE)  # 确保是 [1, C, H, W]
    output = model(img)
    _, pred = torch.max(output, 1)

    print(f'📷 图像: {dataset_test.imgs[index]}, 预测类别: {classes[pred.item()]}')
