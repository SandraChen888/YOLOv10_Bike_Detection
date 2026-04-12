import os

def unify_label_format(label_dir):
    """统一标注文件格式，将类别ID转换为整数"""
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    updated_count = 0
    
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                # 将类别ID转换为整数
                cls_id = int(float(parts[0]))
                parts[0] = str(cls_id)
                updated_lines.append(' '.join(parts))
        
        # 写回更新后的标注文件
        with open(label_path, 'w') as f:
            f.write('\n'.join(updated_lines) + '\n')
        
        updated_count += 1
    
    print(f"更新了 {updated_count} 个标注文件的格式")

# 统一自行车标注格式
print("统一自行车训练集标注格式...")
bike_train_dir = "F:/dataset/bike_dataset/labels/train"
unify_label_format(bike_train_dir)

print("统一自行车验证集标注格式...")
bike_val_dir = "F:/dataset/bike_dataset/labels/val"
unify_label_format(bike_val_dir)

print("统一黄格子训练集标注格式...")
yellow_grid_train_dir = "F:/dataset/yellow_grid_dataset_split/labels/train"
unify_label_format(yellow_grid_train_dir)

print("统一黄格子验证集标注格式...")
yellow_grid_val_dir = "F:/dataset/yellow_grid_dataset_split/labels/val"
unify_label_format(yellow_grid_val_dir)

print("\n标注文件格式统一完成！")
