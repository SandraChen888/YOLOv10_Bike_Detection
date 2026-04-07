import warnings

# 屏蔽 libpng 警告
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
import os

# ====================== 改成你要验证的路径 ======================
# 验证自己标注数据时用：
IMAGE_DIR = r"F:\dataset\clean_bike_images"  # 你的原始图片文件夹
LABEL_DIR = r"F:\dataset\clean_bike_labels"  # 你的原始标签文件夹

# 验证网上数据集时再改成对应路径，例如：
# IMAGE_DIR = r"F:\dataset\clean_bike_images"
# LABEL_DIR = r"F:\dataset\clean_bike_labels"
# ==============================================================

# 遍历所有图片，逐个检查标签
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(IMAGE_DIR, img_name)
    txt_path = os.path.join(LABEL_DIR,
                            img_name.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))

    if not os.path.exists(txt_path):
        print(f"⚠️  无标签: {img_name}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  图片损坏: {img_name}")
        continue
    h, w = img.shape[:2]

    with open(txt_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            print(f"⚠️  空标签: {img_name}")
            continue
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"⚠️  标签格式错误: {img_name} → {line}")
                continue
            cls, x, y, bw, bh = map(float, parts)
            # 检查坐标是否在 0~1 范围内
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                print(f"⚠️  坐标越界: {img_name} → {x} {y} {bw} {bh}")
                continue

    # ========== 弹窗显示代码已注释 ==========
    # 想可视化验证时，再取消下面这段的注释
    x1 = int((x - bw/2) * w)
    y1 = int((y - bh/2) * h)
    x2 = int((x + bw/2) * w)
    y2 = int((y + bh/2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow(f"标注验证: {img_name}", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
         break
    # cv2.destroyAllWindows()
    # =======================================

print("✅ 标签检查完成！")
# cv2.destroyAllWindows()  # 弹窗代码注释后，这行也可以注释掉