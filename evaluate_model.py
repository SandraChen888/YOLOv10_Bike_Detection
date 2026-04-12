from ultralytics import YOLO
import os
import cv2
import numpy as np

# 加载模型
model = YOLO("runs/detect/train_new/weights/best.pt")

# 评估数据集路径
val_images_dir = "F:/dataset/yellow_grid_dataset_split/images/val"

# 类别名称
class_names = {
    0: "bicycle",
    1: "yellow_grid"
}

# 评估结果
results = {
    "total_images": 0,
    "detected_bicycles": 0,
    "detected_yellow_grids": 0,
    "total_confidence": 0
}

# 遍历验证集图片
image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

print("开始评估模型...")
print(f"验证集图片数量: {len(image_files)}")

for i, img_file in enumerate(image_files):
    img_path = os.path.join(val_images_dir, img_file)
    
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"跳过无法读取的图片: {img_file}")
        continue
    
    # 执行检测
    detect_results = model(img, conf=0.5)
    
    # 统计结果
    results["total_images"] += 1
    
    # 解析检测结果
    for res in detect_results[0].boxes:
        cls_id = int(res.cls.cpu().numpy()[0])
        conf = float(res.conf.cpu().numpy()[0])
        
        results["total_confidence"] += conf
        
        if cls_id == 0:
            results["detected_bicycles"] += 1
        elif cls_id == 1:
            results["detected_yellow_grids"] += 1
    
    # 显示进度
    if (i + 1) % 10 == 0:
        print(f"处理进度: {i + 1}/{len(image_files)}")

# 计算平均置信度
avg_confidence = results["total_confidence"] / (results["detected_bicycles"] + results["detected_yellow_grids"])
if (results["detected_bicycles"] + results["detected_yellow_grids"]) == 0:
    avg_confidence = 0

# 打印评估结果
print("\n模型评估结果：")
print(f"验证集图片总数: {results['total_images']}")
print(f"检测到的自行车数量: {results['detected_bicycles']}")
print(f"检测到的黄格子数量: {results['detected_yellow_grids']}")
print(f"平均置信度: {avg_confidence:.4f}")
print(f"每张图片平均检测目标数: {((results['detected_bicycles'] + results['detected_yellow_grids']) / results['total_images']):.2f}")

# 保存评估结果
with open("model_evaluation.txt", "w") as f:
    f.write("模型评估结果：\n")
    f.write(f"验证集图片总数: {results['total_images']}\n")
    f.write(f"检测到的自行车数量: {results['detected_bicycles']}\n")
    f.write(f"检测到的黄格子数量: {results['detected_yellow_grids']}\n")
    f.write(f"平均置信度: {avg_confidence:.4f}\n")
    f.write(f"每张图片平均检测目标数: {((results['detected_bicycles'] + results['detected_yellow_grids']) / results['total_images']):.2f}\n")

print("\n评估完成！结果已保存到 model_evaluation.txt")
