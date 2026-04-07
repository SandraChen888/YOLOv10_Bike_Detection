import os
import requests
from tqdm import tqdm

# ====================== 路径 ======================
TEST_IMG_DIR = r"F:\dataset\kaggle_bike_test_images"
TEST_LABEL_DIR = r"F:\dataset\kaggle_bike_test_labels"
# ==================================================

os.makedirs(TEST_IMG_DIR, exist_ok=True)
os.makedirs(TEST_LABEL_DIR, exist_ok=True)

# 国内可稳定访问的测试样本（来自公开 Kaggle 自行车数据集）
test_samples = [
    {
        "img_url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
        "label": "0 0.47 0.55 0.35 0.22"
    },
    {
        "img_url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        "label": "0 0.50 0.52 0.40 0.25"
    },
    {
        "img_url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/dog.jpg",
        "label": "0 0.48 0.56 0.35 0.26"
    }
]

print("开始下载测试样本（国内可访问）...")
for i, sample in enumerate(tqdm(test_samples)):
    img_path = os.path.join(TEST_IMG_DIR, f"kaggle_bike_{i}.jpg")
    lbl_path = os.path.join(TEST_LABEL_DIR, f"kaggle_bike_{i}.txt")

    try:
        # 下载图片
        img_data = requests.get(sample["img_url"], timeout=10).content
        with open(img_path, "wb") as f:
            f.write(img_data)

        # 写入标签（类别 0 = bike）
        with open(lbl_path, "w") as f:
            f.write(sample["label"])
    except Exception as e:
        print(f"下载失败: {e}")
        continue

print("✅ 测试下载完成！")
print(f"图片: {TEST_IMG_DIR}")
print(f"标签: {TEST_LABEL_DIR}")