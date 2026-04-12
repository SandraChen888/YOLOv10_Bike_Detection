import cv2
import numpy as np

def detect_yellow_grids(image):
    """使用颜色检测黄格子"""
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 黄色的HSV范围（可根据实际情况调整）
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # 阈值化
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤轮廓（根据面积和形状）
    yellow_grids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # 最小面积阈值
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.8 < aspect_ratio < 1.2:  # 接近正方形
                yellow_grids.append((x, y, x+w, y+h))
    
    return yellow_grids

# 测试颜色检测
def test_color_detection(image_path):
    """测试颜色检测"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    grids = detect_yellow_grids(image)
    print(f"检测到 {len(grids)} 个黄格子")
    
    # 绘制检测结果
    for (x1, y1, x2, y2) in grids:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # 保存结果
    output_path = image_path.replace('.jpg', '_color_detection.jpg')
    cv2.imwrite(output_path, image)
    print(f"结果保存到: {output_path}")

# 测试示例
if __name__ == "__main__":
    test_image = "F:/dataset/yellow_grid_enhanced/images/train/enhanced_yellow_grid_0001.jpg"
    test_color_detection(test_image)
