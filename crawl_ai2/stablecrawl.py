import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 配置 Selenium 选项
options = webdriver.ChromeOptions()
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")

# 启动 Chrome 浏览器
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 访问本地网页
driver.get("http://127.0.0.1:7860/")
time.sleep(40)  # 等待页面加载

# 等待按钮出现
wait = WebDriverWait(driver, 10)

try:
    # **使用 ID 定位按钮**
    generate_button = wait.until(EC.element_to_be_clickable((By.ID, "txt2img_generate")))
    print("✅ 找到 'Generate' 按钮")

    # **循环点击 "Generate" 按钮**
    for i in range(200):  # 生成 x00 张图片
        generate_button = wait.until(EC.element_to_be_clickable((By.ID, "txt2img_generate")))
        generate_button.click()
        print(f"🖼️ 第 {i+1} 张图片生成中...")
        time.sleep(10)  # 等待图片生成完成

except Exception as e:
    print(f"❌ 发生错误: {e}")

# **不会关闭你的浏览器**
driver.quit()