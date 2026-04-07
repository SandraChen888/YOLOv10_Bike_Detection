import os

# Kaggle 数据集标签路径
KAGGLE_LABEL_DIR = r"F:\dataset\bicycle\labels"

# 统计所有标签类别
class_counts = {}

for txt_file in os.listdir(KAGGLE_LABEL_DIR):
    if not txt_file.endswith(".txt"):
        continue
    txt_path = os.path.join(KAGGLE_LABEL_DIR, txt_file)
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            # 取第一个数字（类别ID）
            cls_id = line.strip().split()[0]
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

print("📊 Kaggle 数据集标签类别统计：")
for cls, cnt in class_counts.items():
    print(f"类别 {cls}：{cnt} 个标注")