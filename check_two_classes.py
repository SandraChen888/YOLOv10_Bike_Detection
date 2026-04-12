import os

# 你的数据集路径
DATASET_ROOT = r"F:\dataset\merged_bicycle_split"

splits = ["train", "val", "test"]

for split in splits:
    label_dir = os.path.join(DATASET_ROOT, split, "labels")
    has_0 = False
    has_1 = False

    if not os.path.exists(label_dir):
        print(f"{split}: 目录不存在")
        continue

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(label_dir, file)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            c = int(line.split()[0])
            if c == 0:
                has_0 = True
            if c == 1:
                has_1 = True
        if has_0 and has_1:
            break

    print(f"=== {split} ===")
    print(f"类别 0 (bike): {'✅ 存在' if has_0 else '❌ 缺失'}")
    print(f"类别 1 (yellow_grid): {'✅ 存在' if has_1 else '❌ 缺失'}")
    print()