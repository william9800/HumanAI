import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# 配置 Selenium 选项
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # 关闭无头模式，看看是否正常
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")  # 规避反爬

# 可选：使用代理（如果 IP 被封）
# proxy = "http://your_proxy_address:port"
# options.add_argument(f"--proxy-server={proxy}")

# 启动 Chrome 浏览器
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 关键词搜索
search_query = "asian child"
search_url = f"https://lexica.art/?q={search_query.replace(' ', '+')}"

# 打开搜索页面
driver.get(search_url)
time.sleep(5)  # 等待初始加载

# 确保保存目录存在
save_dir = "lexica_child_faces"
os.makedirs(save_dir, exist_ok=True)

# 滚动页面以加载更多图片
def scroll_down():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)  # 等待新图片加载

download_count = 0  # 计数器
max_images = 1000  # 目标下载数量

while download_count < max_images:
    scroll_down()  # 每次滚动加载更多内容
    image_elements = driver.find_elements(By.TAG_NAME, "img")  # 重新获取图片元素

    for img in image_elements:
        if download_count >= max_images:  # 达到目标数量，停止下载
            break

        img_url = img.get_attribute("src")

        # 过滤掉无效 URL（如 base64 编码的图片）
        if not img_url or img_url.startswith("data:image"):
            continue

        try:
            # 下载图片
            img_data = requests.get(img_url, timeout=10).content
            file_path = os.path.join(save_dir, f"face_{download_count}.jpg")

            # 保存图片
            with open(file_path, "wb") as f:
                f.write(img_data)

            print(f"✅ 下载成功: {file_path}")
            download_count += 1

        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败: {img_url}, 错误: {e}")

# 关闭 Selenium
driver.quit()
print("🎉 下载完成！")
