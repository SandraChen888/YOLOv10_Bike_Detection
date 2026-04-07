import os
import shutil
import random
from pathlib import Path

# ====================== 路径配置（完全对应你当前的文件夹）======================
# 你自己的原始数据
ORIGIN_IMG = Path(r"F:\dataset\images")
ORIGIN_LABEL = Path(r"F:\dataset\labels")

# 你自己的增强数据
AUG_IMG = Path(r"F:\dataset\aug_images")
AUG_LABEL = Path(r"F:\dataset\aug_labels")

# Kaggle 公开数据（已统一标签为 0）
KAGGLE_IMG = Path(r"F:\dataset\bicycle\images")
KAGGLE_LABEL = Path(r"F:\dataset\bicycle\labels")

# 整合后输出目录
OUTPUT_ROOT = Path(r"F:\dataset\bike_dataset")
TRAIN_IMG = OUTPUT_ROOT / "images" / "train"
TRAIN_LABEL = OUTPUT_ROOT / "labels" / "train"
VAL_IMG = OUTPUT_ROOT / "images" / "val"
VAL_LABEL = OUTPUT_ROOT / "labels" / "val"

# ====================== 创建输出目录 ======================
for dir_path in [TRAIN_IMG, TRAIN_LABEL, VAL_IMG, VAL_LABEL]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ====================== 收集所有有效数据对 ======================
all_pairs = []

# 收集原始数据
for img_path in ORIGIN_IMG.glob("*.[jp][pn]g"):
    label_path = ORIGIN_LABEL / f"{img_path.stem}.txt"
    if label_path.exists():
        all_pairs.append( (img_path, label_path, f"origin_{img_path.name}") )

# 收集增强数据
for img_path in AUG_IMG.glob("*.[jp][pn]g"):
    label_path = AUG_LABEL / f"{img_path.stem}.txt"
    if label_path.exists():
        all_pairs.append( (img_path, label_path, f"aug_{img_path.name}") )

# 收集 Kaggle 数据
for img_path in KAGGLE_IMG.glob("*.[jp][pn]g"):
    label_path = KAGGLE_LABEL / f"{img_path.stem}.txt"
    if label_path.exists():
        all_pairs.append( (img_path, label_path, f"kaggle_{img_path.name}") )

print(f"✅ 共收集到 {len(all_pairs)} 组有效数据（图片+标签）")

# ====================== 8:2 划分训练/验证集 ======================
random.shuffle(all_pairs)
split_ratio = 0.8
split_idx = int(len(all_pairs) * split_ratio)
train_pairs = all_pairs[:split_idx]
val_pairs = all_pairs[split_idx:]

print(f"✅ 训练集: {len(train_pairs)} 组，验证集: {len(val_pairs)} 组")

# ====================== 复制文件到对应目录 ======================
def copy_files(pairs, img_dst_dir, label_dst_dir):
    for src_img, src_label, new_name in pairs:
        # 复制图片，加前缀防重名
        dst_img = img_dst_dir / new_name
        shutil.copy2(src_img, dst_img)
        # 复制对应标签，重命名匹配图片
        dst_label = label_dst_dir / f"{Path(new_name).stem}.txt"
        shutil.copy2(src_label, dst_label)

# 复制训练集
copy_files(train_pairs, TRAIN_IMG, TRAIN_LABEL)
# 复制验证集
copy_files(val_pairs, VAL_IMG, VAL_LABEL)

print("✅ 数据集整合+划分完成！")
print(f"训练集路径: {TRAIN_IMG}")
print(f"验证集路径: {VAL_IMG}")