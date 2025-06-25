import matplotlib.pyplot as plt

# 数据
x_labels = ['5', '10', '20', '30', '40', '60', '80','100']

data = {
    'EfficientNet-B7 with multi date': [100.00, 100.00, 99.10, 96.60, 90.61, 66.83, 29.27,9.39],
    'EfficientNet-B7 with data processing': [97.1,90.51,80.22,63.34,23.68,8.19,0.10,0.10],
    'EfficientNet-B7': [91.71,65.03,26.07,5.29,0.60,0.10,0.10,0.10],
    'VGG-16': [92.51,25.17,0.10,0.10,0.10,0.10,0.10,0.10],
    'ResNet-50': [99.50,86.51,8.69,0.20,0.10,0.10,0.10,0.10]
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
plt.title('Comparison of detection performance of each model under different Gauss Noise (using 1000 Stylegan+1FFHQ)', fontsize=16)
plt.xlabel('Gauss Noise Level', fontsize=14)
plt.ylabel('Detection accuracy (%)', fontsize=14)
plt.ylim(0, 110)  # Y轴范围稍微高于100

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# 保存 & 显示
plt.savefig('compression_accuracy_curve.png', dpi=300)
plt.show()
