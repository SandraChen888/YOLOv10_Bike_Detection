import yaml
import os

# YAML 文件路径
YAML_PATH = r"F:\dataset\bike_dataset\bicycle.yaml"

# 1. 自动创建 yaml 文件（如果不存在）
if not os.path.exists(YAML_PATH):
    print(f"⚠️ 未找到 {YAML_PATH}，正在自动创建...")
    yaml_content = {
        "path": "F:/dataset/bike_dataset",
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["bicycle"]
    }
    with open(YAML_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ 已自动创建 {YAML_PATH}")

# 2. 读取并检查 YAML
with open(YAML_PATH, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

print("\n🔍 正在检查 YAML 是否正确...\n")

# 检查路径
root_path = data['path']x
train_path = os.path.join(root_path, data['train'])
val_path = os.path.join(root_path, data['val'])

print(f"数据集根目录: {root_path} → {'✅ 存在' if os.path.exists(root_path) else '❌ 错误！'}")
print(f"训练集路径: {train_path} → {'✅ 存在' if os.path.exists(train_path) else '❌ 错误！'}")
print(f"验证集路径: {val_path} → {'✅ 存在' if os.path.exists(val_path) else '❌ 错误！'}")

# 检查类别
nc = data['nc']
print(f"\n类别数量 nc: {nc} → {'✅ 正确' if nc == 1 else '❌ 应该为 1'}")
names = data['names']
print(f"类别名称 names: {names} → {'✅ 正确' if len(names) == 1 else '❌ 长度必须为 1'}")

print("\n🎉 YAML 检查完成！全部正确即可开始训练！")