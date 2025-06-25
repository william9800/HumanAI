import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# 准备数据
data = {

}

df = pd.DataFrame(data)

# 设置可视化风格
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.titlesize': 16
})

# ========= 图 1 =========
fig1, ax1 = plt.subplots(figsize=(12, 6))
top_datasets = df.head(6)
x = np.arange(len(top_datasets))
width = 0.15
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, col in enumerate(df.columns[1:]):
    ax1.bar(x + i*width, top_datasets[col], width, label=col, color=colors[i],
            edgecolor='white', linewidth=0.5)

    # 添加数值标签
    for j, val in enumerate(top_datasets[col]):
        if val > 5:
            ax1.text(x[j] + i*width, val + 1, f'{val:.1f}%',
                     ha='center', va='bottom', fontsize=9, rotation=45)

ax1.set_xticks(x + width*2)
ax1.set_xticklabels(top_datasets["Data Combination"], rotation=45, ha='right')
ax1.set_ylim(0, 110)
ax1.yaxis.set_major_formatter(PercentFormatter())
ax1.set_title('A. 高质量数据组合下的模型表现', pad=20)
ax1.legend(loc='upper right', ncol=3, bbox_to_anchor=(1, 1))
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison_partA.png', dpi=300)
plt.close()

# ========= 图 2 =========
fig2, ax2 = plt.subplots(figsize=(12, 6))
x_full = np.arange(len(df))
for i, col in enumerate(['EB7+MultiData', 'EB7+Preprocess', 'EB7 Base']):
    ax2.bar(x_full + i*width, df[col], width, label=col, color=colors[i],
            edgecolor='white', linewidth=0.5)

    for j, val in enumerate(df[col]):
        if val > 99:
            ax2.plot(x_full[j] + i*width, val + 3, 'r*', markersize=10)

ax2.set_xticks(x_full + width)
ax2.set_xticklabels(df["Data Combination"], rotation=45, ha='right')
ax2.set_ylim(0, 110)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.set_title('B. 不同数据组合下EfficientNet变体对比', pad=20)
ax2.legend(loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison_partB.png', dpi=300)
plt.close()

# ========= 图 3 =========
fig3, ax3 = plt.subplots(figsize=(14, 3))
diff = df[['EB7+MultiData', 'EB7+Preprocess']].sub(df['EB7 Base'], axis=0)
im = ax3.imshow(diff.T, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)

ax3.set_xticks(np.arange(len(df)))
ax3.set_xticklabels(df["Data Combination"], rotation=45, ha='right')
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['+MultiData', '+Preprocess'])
ax3.set_title('C. 改进版相对于基础版的准确率提升幅度(%)', pad=20)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.2)
cbar.set_label('Accuracy Improvement')

plt.tight_layout()
plt.savefig('model_comparison_partC.png', dpi=300)
plt.close()
