from ultralytics import YOLO
import os
import cv2

# 加载模型
model = YOLO("runs/detect/train_new/weights/best.pt")

# 测试图片目录（选择包含自行车的图片）
test_images_dir = "F:/dataset/yellow_grid_dataset_split/images/train"

# 类别名称
class_names = {
    0: "bicycle",
    1: "yellow_grid"
}

# 测试结果
results = {
    "total_images": 0,
    "detected_bicycles": 0,
    "detected_yellow_grids": 0,
    "bicycle_confidence": 0,
    "yellow_grid_confidence": 0
}

# 遍历测试图片
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

print("开始测试自行车检测...")
print(f"测试图片数量: {len(image_files)}")

# 限制测试图片数量（避免处理太多图片）
test_files = image_files[:50]  # 只测试前50张图片

for i, img_file in enumerate(test_files):
    img_path = os.path.join(test_images_dir, img_file)
    
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
        
        if cls_id == 0:
            results["detected_bicycles"] += 1
            results["bicycle_confidence"] += conf
        elif cls_id == 1:
            results["detected_yellow_grids"] += 1
            results["yellow_grid_confidence"] += conf
    
    # 显示进度
    if (i + 1) % 10 == 0:
        print(f"处理进度: {i + 1}/{len(test_files)}")

# 计算平均置信度
bicycle_avg_conf = results["bicycle_confidence"] / results["detected_bicycles"] if results["detected_bicycles"] > 0 else 0
yellow_grid_avg_conf = results["yellow_grid_confidence"] / results["detected_yellow_grids"] if results["detected_yellow_grids"] > 0 else 0

# 打印测试结果
print("\n自行车检测测试结果：")
print(f"测试图片总数: {results['total_images']}")
print(f"检测到的自行车数量: {results['detected_bicycles']}")
print(f"检测到的黄格子数量: {results['detected_yellow_grids']}")
print(f"自行车平均置信度: {bicycle_avg_conf:.4f}")
print(f"黄格子平均置信度: {yellow_grid_avg_conf:.4f}")
print(f"每张图片平均检测到的自行车数: {results['detected_bicycles'] / results['total_images']:.2f}")
print(f"每张图片平均检测到的黄格子数: {results['detected_yellow_grids'] / results['total_images']:.2f}")

# 保存测试结果
with open("bicycle_detection_test.txt", "w") as f:
    f.write("自行车检测测试结果：\n")
    f.write(f"测试图片总数: {results['total_images']}\n")
    f.write(f"检测到的自行车数量: {results['detected_bicycles']}\n")
    f.write(f"检测到的黄格子数量: {results['detected_yellow_grids']}\n")
    f.write(f"自行车平均置信度: {bicycle_avg_conf:.4f}\n")
    f.write(f"黄格子平均置信度: {yellow_grid_avg_conf:.4f}\n")
    f.write(f"每张图片平均检测到的自行车数: {results['detected_bicycles'] / results['total_images']:.2f}\n")
    f.write(f"每张图片平均检测到的黄格子数: {results['detected_yellow_grids'] / results['total_images']:.2f}\n")

print("\n测试完成！结果已保存到 bicycle_detection_test.txt")
