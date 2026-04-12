import os

# 你的数据集路径
LABEL_ROOT = r"F:\dataset\merged_bicycle_dataset\labels"
TRAIN_LABEL = os.path.join(LABEL_ROOT, "train")
VAL_LABEL = os.path.join(LABEL_ROOT, "val")

def count_images_with_class(label_dir, target_class=1):
    image_count = 0
    total_files = 0

    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        total_files += 1
        path = os.path.join(label_dir, f)

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    cls = int(line.split()[0])
                    if cls == target_class:
                        image_count += 1
                        break
        except:
            continue

    return image_count, total_files

# 统计
train_yellow_imgs, train_total = count_images_with_class(TRAIN_LABEL, 1)
val_yellow_imgs, val_total = count_images_with_class(VAL_LABEL, 1)

total_yellow_images = train_yellow_imgs + val_yellow_imgs
total_all_images = train_total + val_total

print("=== 🟡 包含黄格子的图片数量 ===")
print(f"训练集：{train_yellow_imgs} 张 / 共 {train_total} 张")
print(f"验证集：{val_yellow_imgs} 张 / 共 {val_total} 张")
print(f"✅ 总计：{total_yellow_images} 张图片有黄格子")
print(f"占全部图片比例：{total_yellow_images/total_all_images*100:.1f}%")