from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# 配置 Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # 无界面模式
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# 启动浏览器
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 创建保存目录
save_dir = "E:\crawl_data\AI_faces"
os.makedirs(save_dir, exist_ok=True)

for i in range(5000):  # 这里先下载 10 张测试
    driver.get("https://thispersondoesnotexist.com/")  # 打开网站
    time.sleep(3)  # 等待页面加载

    # 获取图片元素
    img_element = driver.find_element("tag name", "img")
    img_url = img_element.get_attribute("src")

    # 下载图片
    img_data = driver.execute_script("return fetch(arguments[0]).then(res => res.blob()).then(blob => blob.arrayBuffer()).then(buf => new Uint8Array(buf));", img_url)
    with open(f"{save_dir}/face_{i}.jpg", "wb") as f:
        f.write(bytearray(img_data))

    print(f"✅ 下载成功: face_{i}.jpg")

driver.quit()
