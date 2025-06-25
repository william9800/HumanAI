import os
import time
import requests
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 🔹 **你的 Chrome 个人资料路径（替换成你的路径！）**
chrome_profile_path = r"C:\Users\86134\AppData\Local\Google\Chrome\User Data"

# 配置 Selenium 选项
options = webdriver.ChromeOptions()
options.add_argument(f"--user-data-dir={chrome_profile_path}")  # ✅ 继承已登录状态
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")  # 规避反爬

# 启动 Chrome 浏览器
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 访问豆包 AI 生成页面
driver.get("https://www.doubao.com/chat/create-image")
time.sleep(5)  # 等待页面加载

# 确保保存目录存在
save_dir = "doubao_images"
os.makedirs(save_dir, exist_ok=True)

# 生成多少张图片
total_images = 1000
batch_size = 4  # 每次 AI 生成 4 张
num_batches = total_images // batch_size

# 等待加载
wait = WebDriverWait(driver, 10)

def download_image(img_element, idx):
    """鼠标悬停并点击下载按钮"""
    try:
        actions = ActionChains(driver)
        actions.move_to_element(img_element).perform()
        time.sleep(1)  # 等待悬停后按钮出现

        # 找到“下载原图”按钮
        download_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//div[@data-testid='edit_image_hover_tag_download_btn']"))
        )
        download_button.click()
        time.sleep(3)  # 等待下载弹窗出现

        # 获取原图链接
        img_url = img_element.get_attribute("src")
        img_data = requests.get(img_url, timeout=10).content
        file_path = os.path.join(save_dir, f"image_{idx}.jpg")

        with open(file_path, "wb") as f:
            f.write(img_data)
        print(f"✅ 下载完成: {file_path}")

    except Exception as e:
        print(f"❌ 下载失败: {e}")

for batch in range(num_batches):
    print(f"\n🚀 生成第 {batch + 1} 批图片...")
    try:
        input_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[contenteditable='true']")))
        input_box.click()
        time.sleep(1)
        input_box.send_keys(Keys.CONTROL + "a")
        input_box.send_keys(Keys.DELETE)
        time.sleep(1)
        input_text = "生成一个中国人的图片"
        input_box.send_keys(input_text)
        time.sleep(1)
        send_button = wait.until(EC.element_to_be_clickable((By.ID, "flow-end-msg-send")))
        send_button.click()
        print("✅ 发送按钮点击成功！")
        time.sleep(10)

        # 获取所有生成的图片
        images = driver.find_elements(By.XPATH, "//img[contains(@src, 'image')]")
        if not images:
            print("❌ 没有找到 AI 生成的图片")
            continue

        print(f"✅ 找到 {len(images)} 张图片，开始下载...")
        for idx, img in enumerate(images[:batch_size]):
            download_image(img, batch * batch_size + idx)

    except Exception as e:
        print(f"⚠️ 生成或下载图片失败: {e}")
    time.sleep(5)