# illegal_judge.py
import cv2
import numpy as np

# 预设校园禁停区域（核心：按图片像素坐标定义，后续根据校园监控/拍摄画面修改）
# 格式：{场景: {违规类型: [[x1,y1,x2,y2], ...]}} （矩形区域，x1y1左上角，x2y2右下角）
scene_illegal_areas = {
    "教学楼A栋": {
        "人行道": [[100, 50, 500, 800]],
        "教学楼门口": [[600, 100, 900, 700]],
        "消防通道": [[200, 200, 700, 600]],
        "绿化带": [[0, 300, 400, 900]]
    },
    "教学楼B栋": {
        "人行道": [[100, 50, 500, 800]],
        "教学楼门口": [[600, 100, 900, 700]],
        "消防通道": [[200, 200, 700, 600]],
        "绿化带": [[0, 300, 400, 900]]
    },
    "教学楼C栋": {
        "人行道": [[100, 50, 500, 800]],
        "教学楼门口": [[600, 100, 900, 700]],
        "消防通道": [[200, 200, 700, 600]],
        "绿化带": [[0, 300, 400, 900]]
    },
    "教学楼D栋": {
        "人行道": [[100, 50, 500, 800]],
        "教学楼门口": [[600, 100, 900, 700]],
        "消防通道": [[200, 200, 700, 600]],
        "绿化带": [[0, 300, 400, 900]]
    },
    "教学楼E栋": {
        "人行道": [[100, 50, 500, 800]],
        "教学楼门口": [[600, 100, 900, 700]],
        "消防通道": [[200, 200, 700, 600]],
        "绿化带": [[0, 300, 400, 900]]
    },
    "教学楼F栋": {
        "人行道": [[100, 50, 500, 800]],
        "教学楼门口": [[600, 100, 900, 700]],
        "消防通道": [[200, 200, 700, 600]],
        "绿化带": [[0, 300, 400, 900]]
    },
    "一饭堂门口": {
        "人行道": [[100, 50, 500, 800]],
        "教学楼门口": [[600, 100, 900, 700]],
        "消防通道": [[200, 200, 700, 600]],
        "绿化带": [[0, 300, 400, 900]]
    }
}

# 默认禁停区域（用于兼容旧代码）
illegal_areas = scene_illegal_areas["教学楼A栋"]


def calculate_overlap(box1, box2):
    """
    计算两个矩形的重叠面积占比（box：[x1,y1,x2,y2]）
    :return: 重叠面积占检测框的比例
    """
    # 计算交集坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 无交集
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # 计算面积
    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return inter_area / box1_area if box1_area > 0 else 0.0


def judge_illegal(detect_res, img, overlap_thresh=0.3, scene="教学楼A栋"):
    """
    违规判定核心函数
    :param detect_res: 检测结果（来自bike_detect）
    :param img: 检测后的图片
    :param overlap_thresh: 重叠阈值（≥30%则判定违规）
    :param scene: 当前场景
    :return: 违规结果（列表）、标记违规后的图片
    """
    illegal_res = []
    img_copy = img.copy()

    # 分离自行车和黄格子检测结果
    bikes = []
    yellow_grids = []
    for cls_id, conf, x1, y1, x2, y2 in detect_res:
        if cls_id == 0:  # 自行车
            bikes.append((conf, x1, y1, x2, y2))
        elif cls_id == 1:  # 黄格子
            yellow_grids.append((x1, y1, x2, y2))

    # 获取当前场景的禁停区域
    current_illegal_areas = scene_illegal_areas.get(scene, illegal_areas)

    # 遍历每个检测到的单车
    for idx, (conf, x1, y1, x2, y2) in enumerate(bikes):
        bike_box = [x1, y1, x2, y2]
        is_illegal = False
        illegal_type = ""
        max_overlap = 0.0

        # 检查是否在黄格子内
        in_yellow_grid = False
        for grid_box in yellow_grids:
            overlap = calculate_overlap(bike_box, grid_box)
            if overlap >= 0.5:  # 自行车50%以上在黄格子内视为合规
                in_yellow_grid = True
                break

        # 如果不在黄格子内，检查是否在禁停区域
        if not in_yellow_grid:
            # 遍历每个禁停区域，判断是否重叠
            for area_type, areas in current_illegal_areas.items():
                for area in areas:
                    overlap = calculate_overlap(bike_box, area)
                    if overlap >= overlap_thresh and overlap > max_overlap:
                        is_illegal = True
                        illegal_type = area_type
                        max_overlap = overlap
        else:
            # 在黄格子内，视为合规
            is_illegal = False

        # 标记违规结果
        if is_illegal:
            illegal_res.append({
                "单车编号": idx + 1,
                "置信度": conf,
                "检测框": bike_box,
                "违规类型": illegal_type,
                "重叠占比": max_overlap
            })
            # 画违规框（黄色，线宽3）+ 标注违规类型
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(img_copy, f"Illegal: {illegal_type}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            # 合规单车（绿色框）
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, "Legal", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 画禁停区域轮廓（浅蓝色，线宽1，透明）
    for area_type, areas in current_illegal_areas.items():
        for area in areas:
            x1_a, y1_a, x2_a, y2_a = area
            cv2.rectangle(img_copy, (x1_a, y1_a), (x2_a, y2_a), (255, 255, 0), 1)
            cv2.putText(img_copy, area_type, (x1_a, y1_a - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # 画黄格子轮廓（黄色，线宽2）
    for grid_box in yellow_grids:
        x1_g, y1_g, x2_g, y2_g = grid_box
        cv2.rectangle(img_copy, (x1_g, y1_g), (x2_g, y2_g), (0, 255, 255), 2)
        cv2.putText(img_copy, "Yellow Grid", (x1_g, y1_g - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return illegal_res, img_copy


# 测试代码
if __name__ == "__main__":
    # 模拟检测结果
    test_detect = [[1, 0.85, 200, 300, 300, 450]]
    test_img = cv2.imread("detect_result.jpg")
    if test_img is not None:
        illegal_res, mark_img = judge_illegal(test_detect, test_img)
        print(f"违规判定结果：{illegal_res}")
        cv2.imwrite("illegal_mark.jpg", mark_img)
        print("违规标记完成，结果已保存为illegal_mark.jpg")