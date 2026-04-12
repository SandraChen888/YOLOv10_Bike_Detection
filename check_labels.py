import os

# 你的训练集标签路径
label_dir = r"merged_bicycle_split\train\labels"

class1_count = 0
error_lines = []

for txt_file in os.listdir(label_dir):
    path = os.path.join(label_dir, txt_file)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            error_lines.append(f"{txt_file} 第{line_num}行：列数不是5列 → 会被YOLO丢弃")
            continue

        cls, x, y, w, h = parts
        if cls == "1":
            class1_count += 1

        # 检查是否越界
        try:
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                error_lines.append(f"{txt_file} 第{line_num}行：坐标越界 → 被过滤")
        except:
            error_lines.append(f"{txt_file} 第{line_num}行：格式错误 → 被过滤")

print("="*50)
print(f"【检测完成】")
print(f"数据集中 类别1 标注数量：{class1_count}")
print("="*50)

if error_lines:
    print("\n❌ 发现错误标签（这些会被YOLO直接丢掉！）：\n")
    for err in error_lines[:20]:
        print(err)
    print("\n类别1消失的原因就是：这些标注格式不对，被YOLO过滤了！")
else:
    print("\n✅ 标签格式全部正常！")
    print("如果训练 still 只显示 1 class，原因是：.cache 缓存文件没删！")