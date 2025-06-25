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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from dataset.dataset import HumanAI
from efficientnet_pytorch import EfficientNet

# ==== 配置 ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "E:\HumanAI\saved model\实验4/model_final.pth"
BATCH_SIZE = 8
NUM_CLASSES = 2
CLASS_NAMES = ['real', 'fake']

# ==== 预处理 ====
transform = transforms.Compose([
    transforms.CenterCrop(600),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== 数据加载 ====
test_set = HumanAI("E:\HumanAI/data/test", transforms=transform, test=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ==== 模型加载 ====
model = EfficientNet.from_pretrained('efficientnet-b7')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 1)  # BCE 输出
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==== 特征提取函数 ====
def extract_features(model, loader):
    y_true, y_pred, y_scores, features = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            feat = model.extract_features(x)  # ✅ 获取中间特征
            pooled_feat = nn.functional.adaptive_avg_pool2d(feat, (1,1)).squeeze()  # ✅ 全局池化成 (batch_size, feature_dim)

            out = model(x).squeeze(1)
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).int()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_scores.extend(prob.cpu().numpy())

            features.append(pooled_feat.cpu().numpy())  # ✅ 收集中间特征 (batch_size, feature_dim)

    return np.array(y_true), np.array(y_pred), np.array(y_scores), np.vstack(features)

# ==== 获取预测结果 ====
y_true, y_pred, y_scores, feats = extract_features(model, test_loader)

# ✅ 计算准确率
acc = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy: {acc:.2%}")
# ==== 分类报告 ====
print("🎯 分类报告:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ==== 混淆矩阵 ====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ==== ROC 曲线 ====
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

# ==== PR 曲线 ====
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


