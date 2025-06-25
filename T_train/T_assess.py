import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.manifold import TSNE
from dataset.dataset import HumanAI

# 全局配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 加载测试数据
test_set = HumanAI("E:/HumanAI/data/test", transforms=transform, test=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# 类别信息
classes = test_set.classes
num_classes = len(classes)

# 模型定义映射（和训练保持一致）
model_heads = {
    "efficientnet_b7": lambda m: setattr(m, 'classifier', nn.Sequential(m.classifier[0], nn.Linear(m.classifier[1].in_features, num_classes))),
    "vgg16": lambda m: setattr(m, 'classifier', nn.Sequential(*list(m.classifier.children())[:-1], nn.Linear(m.classifier[6].in_features, num_classes))),
    "resnet50": lambda m: setattr(m, 'fc', nn.Linear(m.fc.in_features, num_classes)),
}

# 保存比较结果
results = []

# 加载并评估所有模型
for name, constructor in {"efficientnet_b7": models.efficientnet_b7, "vgg16": models.vgg16, "resnet50": models.resnet50}.items():
    print(f"\n📊 正在评估模型: {name}")
    model = constructor(pretrained=False)
    model_heads[name](model)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"model_{name}.pth"), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    y_true, y_pred, y_score, features = [], [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs[:, 1].cpu().numpy())
            features.append(outputs.cpu().numpy())

    # 分类报告与混淆矩阵
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Accuracy: {acc:.2%}")
    print(classification_report(y_true, y_pred, target_names=classes))
    results.append((name, acc * 100))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"{name.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_roc_curve.png"))
    plt.close()

    # PR 曲线
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.title(f"PR Curve - {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_pr_curve.png"))
    plt.close()

    # t-SNE 可视化
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb = tsne.fit_transform(np.vstack(features))
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=y_true, palette=["blue", "red"], alpha=0.7)
        plt.title(f"t-SNE - {name}")
        plt.legend(title="Class")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_tsne.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ t-SNE 跳过: {e}")

# 条形图比较
import pandas as pd
df = pd.DataFrame(results, columns=["Model", "Accuracy"])
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="Accuracy", y="Model", palette="viridis")
for i, row in df.iterrows():
    plt.text(row.Accuracy + 0.5, i, f"{row.Accuracy:.2f}%", va='center')
plt.title("模型准确率对比")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"))
plt.close()
