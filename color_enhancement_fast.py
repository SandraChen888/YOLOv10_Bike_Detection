import os
import cv2
import numpy as np
import concurrent.futures

def enhance_image(img_path, output_path):
    """增强单张图像"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        # 增强黄色通道
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 增强饱和度和亮度
        s = cv2.add(s, 50)
        v = cv2.add(v, 30)
        
        # 限制值范围
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        
        # 合并通道
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # 保存增强后的图像
        cv2.imwrite(output_path, enhanced_img)
        return True
    except Exception as e:
        print(f"处理 {img_path} 时出错: {e}")
        return False

def enhance_yellow_grid_data(input_dir, output_dir):
    """增强黄格子数据的颜色特征"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    tasks = []
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"enhanced_{img_file}")
        tasks.append((img_path, output_path))
    
    # 使用多线程处理
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda x: enhance_image(x[0], x[1]), tasks))
        success_count = sum(results)
    
    return success_count

# 增强黄格子数据
train_input_dir = "F:/dataset/yellow_grid_dataset_split/images/train"
train_output_dir = "F:/dataset/yellow_grid_enhanced/images/train"
val_input_dir = "F:/dataset/yellow_grid_dataset_split/images/val"
val_output_dir = "F:/dataset/yellow_grid_enhanced/images/val"

# 创建目录
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# 增强训练集
train_success = enhance_yellow_grid_data(train_input_dir, train_output_dir)
print(f"增强了 {train_success} 张训练集图片")

# 增强验证集
val_success = enhance_yellow_grid_data(val_input_dir, val_output_dir)
print(f"增强了 {val_success} 张验证集图片")
