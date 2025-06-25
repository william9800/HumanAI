import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image

# 配置 Selenium 选项
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # 关闭无头模式，看看是否正常
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")  # 规避反爬

# 启动 Chrome 浏览器
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 关键词搜索
search_query = "asian"
search_url = f"https://lexica.art/{search_query.replace(' ', '-')}"

# 打开搜索页面
driver.get(search_url)
time.sleep(5)  # 等待初始加载

# 确保保存目录存在
save_dir = "lexica"
os.makedirs(save_dir, exist_ok=True)

# 记录已下载的图片 URL，防止重复
downloaded_urls = set()

# 滚动页面 + 处理 Load More 按钮
def load_more_images():
    try:
        # 查找 "Load More" 按钮
        load_more_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Load more')]")
        load_more_button.click()
        print("🔄 点击 'Load More' 按钮")
        time.sleep(5)  # 等待加载
    except:
        # 如果找不到 "Load More" 按钮，就滚动页面
        print("📜 滚动页面加载新图片")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)

download_count = 0  # 计数器
max_images = 100  # 目标下载数量

while download_count < max_images:
    load_more_images()  # 触发加载更多图片
    image_elements = driver.find_elements(By.TAG_NAME, "img")  # 重新获取图片元素

    for img in image_elements:
        if download_count >= max_images:  # 达到目标数量，停止下载
            break

        img_url = img.get_attribute("src")

        # 过滤掉无效 URL（如 base64 编码的图片）或重复 URL
        if not img_url or img_url.startswith("data:image") or img_url in downloaded_urls:
            continue

        try:
            # 下载图片
            img_data = requests.get(img_url, timeout=10).content
            file_path = os.path.join(save_dir, f"face_{download_count}.jpg")

            # 先保存图片
            with open(file_path, "wb") as f:
                f.write(img_data)

            # 使用 Pillow 读取图片尺寸
            with Image.open(file_path) as img:
                width, height = img.size

            # 过滤掉不符合尺寸的图片
            if (width, height) in [(1, 1), (3000, 32)]:
                os.remove(file_path)  # 删除图片
                print(f"❌ 过滤掉尺寸 {width}x{height} 的图片: {img_url}")
                continue  # 不计入下载数量

            print(f"✅ 下载成功: {file_path} ({width}x{height})")
            downloaded_urls.add(img_url)  # 记录已下载的图片 URL
            download_count += 1

        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败: {img_url}, 错误: {e}")
        except Exception as e:
            print(f"⚠️ 图片处理错误: {file_path}, 错误: {e}")

# 关闭 Selenium
driver.quit()
print("🎉 下载完成！")
