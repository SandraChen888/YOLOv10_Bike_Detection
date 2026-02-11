# detect_model.py
from ultralytics import YOLO
import cv2
import os

# 初始化YOLOv10模型（先加载官方预训练权重，后续替换为自己训练的权重）
model = YOLO("yolov10s.pt")  # 兼顾精度和速度，适配毕设指标


def bike_detect(image_path, conf=0.5):
    """
    单车目标检测核心函数
    :param image_path: 图片路径
    :param conf: 置信度阈值（低于则不检测）
    :return: 检测结果（列表，每个元素是[类别id, 置信度, x1, y1, x2, y2]）、检测后的图片
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"图片路径错误，无法读取：{image_path}")

    # 执行检测（只检测自行车：COCO中bicycle的类别id是1，后续自己训练后改0）
    results = model(img, classes=[1], conf=conf)  # 临时用COCO的bikeid，自己训练后改为classes=[0]

    # 解析检测结果
    detect_res = []
    for res in results[0].boxes:
        cls_id = int(res.cls.cpu().numpy()[0])  # 类别id
        conf = float(res.conf.cpu().numpy()[0])  # 置信度
        x1, y1, x2, y2 = map(int, res.xyxy.cpu().numpy()[0])  # 检测框左上角/右下角坐标
        detect_res.append([cls_id, conf, x1, y1, x2, y2])
        # 画检测框（红色，线宽2）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 标注置信度
        cv2.putText(img, f"bike {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return detect_res, img


# 测试代码（直接运行该文件可测试）
if __name__ == "__main__":
    # 替换为你自己的测试图片路径（随便找一张有单车的图片即可）
    test_img = "test_bike.jpg"
    try:
        res, det_img = bike_detect(test_img)
        print(f"检测结果：{res}")
        # 保存检测后的图片
        cv2.imwrite("detect_result.jpg", det_img)
        print("检测完成，结果已保存为detect_result.jpg")
    except Exception as e:
        print(f"检测失败：{e}")