import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_accuracy_bar(csv_path="robustness_eval.csv"):
    df = pd.read_csv(csv_path)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Source", y="Accuracy (%)", data=df, palette="coolwarm")
    plt.title("🎯 模型鲁棒性评估")
    plt.ylim(0, 100)
    for i, val in enumerate(df["Accuracy (%)"]):
        plt.text(i, val + 1, f"{val:.1f}%", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("robustness_plot.png")
    plt.show()
