import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.manifold import TSNE
from tqdm import tqdm

from dataset.dataset import HumanAI
from efficientnet_pytorch import EfficientNet

# ==== 配置 ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "E:\HumanAI\saved model\model2\model_final.pth"
BATCH_SIZE = 8
CLASS_NAMES = ['real', 'fake']

# ==== 预处理 ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ==== 数据加载 ====
test_set = HumanAI("E:\HumanAI/data/test", transforms=transform, test=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ==== 模型加载 ====
model = EfficientNet.from_pretrained('efficientnet-b7')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 2)  # 二分类输出2类
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==== 特征提取函数 ====
def extract_features(model, loader):
    y_true, y_pred, y_scores, features = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)  # [batch, 2]
            prob = torch.softmax(out, dim=1)[:, 1]  # 取正类概率
            pred = torch.argmax(out, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_scores.extend(prob.cpu().numpy())

            # 提取特征（用 feature extractor）
            feat = model.extract_features(x)  # [batch, channels, H, W]
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # [batch, channels]
            features.append(feat.cpu().numpy())


    return np.array(y_true), np.array(y_pred), np.array(y_scores), np.vstack(features)

y_true, y_pred, y_scores, feats = extract_features(model, test_loader)

# ✅ 计算准确率
acc = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy: {acc:.2%}")

# ✅ 分类报告
print("🎯 分类报告:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ✅ 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ✅ ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# ✅ PR 曲线
precision, recall, _ = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pr_curve.png")
plt.show()

# ✅ t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(feats)

plt.figure(figsize=(8, 6))
palette_dict = {0: "blue", 1: "red"}
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_true,
                palette=palette_dict, alpha=0.7)
plt.title("t-SNE of Extracted Features")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=CLASS_NAMES, title="True Label")
plt.tight_layout()
plt.savefig("tsne_visualization.png")
plt.show()
