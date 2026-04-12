import os
import shutil
import random
from tqdm import tqdm

# ===================== 你的真实目录结构 =====================
# 根目录：F:\dataset\merged_bicycle_dataset
# 图片路径：F:\dataset\merged_bicycle_dataset\images\train + images\val
# 标签路径：F:\dataset\merged_bicycle_dataset\labels\train + labels\val
# ==========================================================
DATASET_ROOT = r"F:\dataset\merged_bicycle_dataset"
OUTPUT_ROOT = r"F:\dataset\merged_bicycle_split"  # 划分后输出目录

# 划分比例 8:1:1
RATIO_TRAIN = 0.8
RATIO_VAL = 0.1
RATIO_TEST = 0.1

# 支持的图片格式
IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.bmp']

# ---------------------- 1. 自动收集所有图片 ----------------------
# 图片的两个来源文件夹
image_src_folders = [
    os.path.join(DATASET_ROOT, "images", "train"),
    os.path.join(DATASET_ROOT, "images", "val")
]
# 标签的两个来源文件夹（和图片一一对应）
label_src_folders = [
    os.path.join(DATASET_ROOT, "labels", "train"),
    os.path.join(DATASET_ROOT, "labels", "val")
]

# 收集所有图片路径（去重，避免重复）
all_image_paths = []
all_label_paths = []

# 遍历两个图片文件夹
for img_folder, lbl_folder in zip(image_src_folders, label_src_folders):
    if not os.path.exists(img_folder):
        print(f"⚠️ 跳过不存在的图片目录: {img_folder}")
        continue
    if not os.path.exists(lbl_folder):
        print(f"⚠️ 跳过不存在的标签目录: {lbl_folder}")
        continue

    # 遍历文件夹内所有图片
    for filename in os.listdir(img_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXT:
            continue  # 跳过非图片文件

        img_path = os.path.join(img_folder, filename)
        name = os.path.splitext(filename)[0]
        lbl_path = os.path.join(lbl_folder, name + ".txt")

        # 只保留有对应标签的图片
        if os.path.exists(lbl_path):
            all_image_paths.append(img_path)
            all_label_paths.append(lbl_path)

# ---------------------- 2. 检查数据 ----------------------
total = len(all_image_paths)
if total == 0:
    print("❌ 没有找到任何带对应标签的图片！请检查目录结构")
    exit()

print(f"✅ 共收集到 {total} 组 图片+标签")

# ---------------------- 3. 打乱并按8:1:1划分 ----------------------
# 打包图片和标签，一起打乱
data_pairs = list(zip(all_image_paths, all_label_paths))
random.shuffle(data_pairs)  # 随机打乱

# 计算划分数量
train_num = int(total * RATIO_TRAIN)
val_num = int(total * RATIO_VAL)

train_pairs = data_pairs[:train_num]
val_pairs = data_pairs[train_num: train_num + val_num]
test_pairs = data_pairs[train_num + val_num:]

print(f"\n📊 划分结果（8:1:1）：")
print(f"训练集: {len(train_pairs)} 组")
print(f"验证集: {len(val_pairs)} 组")
print(f"测试集: {len(test_pairs)} 组")

# ---------------------- 4. 创建输出目录 ----------------------
# 标准YOLO格式：output/train/images, train/labels, val/..., test/...
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, split, "labels"), exist_ok=True)


# ---------------------- 5. 复制文件到对应集合 ----------------------
def copy_split(pairs, split_name):
    for img_path, lbl_path in tqdm(pairs, desc=f"复制 {split_name} 集"):
        # 提取文件名
        img_filename = os.path.basename(img_path)
        lbl_filename = os.path.basename(lbl_path)

        # 目标路径
        dst_img = os.path.join(OUTPUT_ROOT, split_name, "images", img_filename)
        dst_lbl = os.path.join(OUTPUT_ROOT, split_name, "labels", lbl_filename)

        # 复制图片和标签
        shutil.copy(img_path, dst_img)
        shutil.copy(lbl_path, dst_lbl)


# 执行复制
copy_split(train_pairs, "train")
copy_split(val_pairs, "val")
copy_split(test_pairs, "test")

# ---------------------- 6. 完成提示 ----------------------
print("\n🎉 数据集划分完成！")
print(f"📂 输出目录: {OUTPUT_ROOT}")
print("✅ 目录结构符合YOLO标准，可直接用于训练")