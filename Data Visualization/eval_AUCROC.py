#绘制roc曲线和AUC值用
import torch
import torchvision.transforms as transforms
from dataset.dataset import HumanAI
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设备选择
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 图像预处理
transform_test = transforms.Compose([
    transforms.CenterCrop(600),  # EfficientNet-B7推荐尺寸
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 加载模型
model = EfficientNet.from_pretrained('efficientnet-b7')
num_ftrs = model._fc.in_features
model._fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("E:/HumanAI/model_final.pth", weights_only=True))
model.to(DEVICE)
model.eval()

# 加载测试集
dataset_test = HumanAI('E:/HumanAI/data/test', transforms=transform_test, test=True)
classes = dataset_test.classes
print(f"📌 测试集类别: {classes}")

if len(dataset_test) == 0:
    raise ValueError("❌ 测试集为空，请检查路径！")

# 收集真实标签和预测概率
y_true, y_pred, y_scores = [], [], []

for img, label in dataset_test:
    if img is None: continue
    img = img.unsqueeze(0).to(DEVICE)
    output = model(img)
    prob = torch.softmax(output, dim=1).cpu().detach().numpy()[0]
    pred = np.argmax(prob)

    y_true.append(label)
    y_pred.append(pred)
    y_scores.append(prob[1])  # 针对类别1的概率（通常为 fake）

# 准确率 & 分类报告
acc = accuracy_score(y_true, y_pred)
print(f"✅ 准确率: {acc:.2%}")
print("\n🎯 分类报告:")
print(classification_report(y_true, y_pred, target_names=classes))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 计算 ROC & AUC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
