from ultralytics import YOLO
import cv2
from color_detection import detect_yellow_grids

def calculate_iou(box1, box2):
    """计算IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def fusion_detect(image_path, model_path="runs/detect/train12/weights/best.pt"):
    """融合颜色检测和YOLO检测"""
    # 加载YOLO模型
    model = YOLO(model_path)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return []
    
    # 颜色检测黄格子
    color_grids = detect_yellow_grids(image)
    
    # YOLO检测
    results = model.predict(image_path, conf=0.3)
    
    # 融合结果
    final_results = []
    
    # 处理YOLO检测结果
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            if cls_id == 1:  # 黄格子
                # 检查是否与颜色检测结果匹配
                matched = False
                for (cx1, cy1, cx2, cy2) in color_grids:
                    # 计算IoU
                    iou = calculate_iou((x1, y1, x2, y2), (cx1, cy1, cx2, cy2))
                    if iou > 0.5:
                        matched = True
                        break
                
                if matched or conf > 0.6:  # 颜色匹配或置信度高
                    final_results.append((cls_id, conf, x1, y1, x2, y2))
            else:  # 自行车
                if conf > 0.3:
                    final_results.append((cls_id, conf, x1, y1, x2, y2))
    
    return final_results

# 测试融合检测
def test_fusion_detection(image_path):
    """测试融合检测"""
    results = fusion_detect(image_path)
    print(f"融合检测结果: {len(results)}个目标")
    
    # 绘制检测结果
    image = cv2.imread(image_path)
    class_names = {0: "bicycle", 1: "yellow_grid"}
    class_colors = {0: (0, 0, 255), 1: (0, 255, 255)}
    
    for cls_id, conf, x1, y1, x2, y2 in results:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), class_colors[cls_id], 2)
        cv2.putText(image, f"{class_names[cls_id]} {conf:.2f}", 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_colors[cls_id], 2)
    
    # 保存结果
    output_path = image_path.replace('.jpg', '_fusion_detection.jpg')
    cv2.imwrite(output_path, image)
    print(f"结果保存到: {output_path}")

# 测试示例
if __name__ == "__main__":
    test_image = "F:/dataset/balanced_dataset/images/train/bike_kaggle_1_1.jpg"
    test_fusion_detection(test_image)
