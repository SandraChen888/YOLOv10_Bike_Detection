import os

# ===================== 你的路径 =====================
DATASET_ROOT = r"F:\dataset\merged_bicycle_split"
# ====================================================

# 要检查的类别
TARGET_CLASS = 1

# 三个集合
splits = ["train", "val", "test"]

# 检查结果
result = {}

for split in splits:
    label_dir = os.path.join(DATASET_ROOT, split, "labels")
    has_target = False

    if not os.path.exists(label_dir):
        result[split] = "❌ 标签目录不存在"
        continue

    # 遍历所有标签
    for lbl_file in os.listdir(label_dir):
        if not lbl_file.endswith(".txt"):
            continue

        lbl_path = os.path.join(label_dir, lbl_file)
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if cls_id == TARGET_CLASS:
                has_target = True
                break
        if has_target:
            break

    result[split] = "✅ 包含类别 1" if has_target else "❌ 没有类别 1"

# ===================== 输出结果 =====================
print("\n" + "="*50)
print("        数据集类别 1 检测结果（bicycle）")
print("="*50)

for split, res in result.items():
    print(f"{split:>10}: {res}")

print("="*50)