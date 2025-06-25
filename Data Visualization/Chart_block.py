import matplotlib.pyplot as plt

# 数据
x_labels = ['5', '40', '100', '200', '400', '500', '600']

data = {
    'EfficientNet-B7 with multi date': [100,100,100,99.9,99.4,96.8,78.52],
    'EfficientNet-B7 with data processing': [98.8,97.8,97.2,95.3,71.43,46.15,13.59],
    'EfficientNet-B7': [99.8,99.8,99.1,95.7,68.73,27.77,0.1],
    'VGG-16': [99.5,99.7,98,90.51,48.95,7.89,0.1],
    'ResNet-50': [100,99.1,99.1,92.91,57.14,8.89,0.1]
}

# 颜色列表（可调整）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.figure(figsize=(12, 7))

for i, (model, values) in enumerate(data.items()):
    plt.plot(x_labels, values, marker='o', label=model, color=colors[i])
    # 每个点上标注数值
    for x, y in zip(x_labels, values):
        plt.text(x, y + 1, f'{y:.2f}%', ha='center', va='bottom', fontsize=9)

# 标题与标签
plt.title('Comparison of detection performance of each model under different block occlusion (using 1000 Stylegan+1FFHQ)', fontsize=16)
plt.xlabel('block occlusion', fontsize=14)
plt.ylabel('Detection accuracy (%)', fontsize=14)
plt.ylim(0, 110)  # Y轴范围稍微高于100

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# 保存 & 显示
plt.savefig('occlusion_accuracy_curve.png', dpi=300)
plt.show()
