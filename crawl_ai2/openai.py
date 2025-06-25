import os
import time
import requests

# 你的 OpenAI API Key
API_KEY = "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"

# 目标：生成 1000 张图片，每次生成 4 张
total_images = 1000
batch_size = 4  # 每次请求生成 4 张
num_batches = total_images // batch_size  # 计算需要请求的次数

# 保存图片的目录
save_dir = "openai_generated_images"
os.makedirs(save_dir, exist_ok=True)

# OpenAI API 端点
url = "https://api.openai.com/v1/images/generations"

# 图片描述（可以修改）
prompt = "一个中国人"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 记录已下载的图片数
download_count = 0

for i in range(num_batches):
    print(f"🚀 请求 {i+1}/{num_batches} 批次，正在生成 4 张图片...")

    payload = {
        "prompt": prompt,
        "n": batch_size,  # 一次生成 4 张
        "size": "1024x1024"  # 可选大小："256x256", "512x512", "1024x1024"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # 检查是否有错误
        data = response.json()

        for idx, img_info in enumerate(data["data"]):
            img_url = img_info["url"]  # 获取 AI 生成的图片 URL

            # 下载图片
            img_data = requests.get(img_url, timeout=10).content
            file_path = os.path.join(save_dir, f"ai_image_{download_count}.jpg")

            # 保存图片
            with open(file_path, "wb") as f:
                f.write(img_data)

            print(f"✅ 下载成功: {file_path}")
            download_count += 1

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")

    # 添加延迟，防止 API 速率限制
    time.sleep(3)

print("🎉 1000 张图片已下载完成！")
