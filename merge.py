import os

# Kaggle 数据集标签路径
KAGGLE_LABEL_DIR = r"F:\dataset\bicycle\labels"
# 原标签备份目录（可在修改错误时恢复）
BACKUP_DIR = r"F:\dataset\bicycle\labels_backup"
os.makedirs(BACKUP_DIR, exist_ok=True)

print("🔄 开始统一 Kaggle 标签：将类别 1 改为 0（与你的 bike 标签对齐）...")
modified_count = 0

for txt_file in os.listdir(KAGGLE_LABEL_DIR):
    if not txt_file.endswith(".txt"):
        continue
    txt_path = os.path.join(KAGGLE_LABEL_DIR, txt_file)

    # 1. 备份原文件
    backup_path = os.path.join(BACKUP_DIR, txt_file)
    with open(txt_path, "r", encoding="utf-8") as f:
        original_content = f.read()
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(original_content)

    # 2. 修改标签：把第一个数字从 1 改为 0
    new_lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 仅修改类别ID，其余坐标完全保留
            if parts[0] == "1":
                parts[0] = "0"
                modified_count += 1
            new_lines.append(" ".join(parts) + "\n")

    # 3. 写入修改后的标签
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

print(f"✅ 标签统一完成！共修改 {modified_count} 个标注，所有 Kaggle 标签已改为类别 0")
print(f"原标签已备份至：{BACKUP_DIR}，可随时恢复")