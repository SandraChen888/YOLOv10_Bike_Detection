# 单独检测一张图片并保存结果
import os
from detect_model import bike_detect
from illegal_judge import judge_illegal
from record_manage import save_illegal_record
import cv2

# 用户指定的图片路径
test_img = "D:\\桌面\\微信图片_20260411042959_93_537.jpg"
# 输出目录
output_dir = "D:\\桌面"

# 执行检测
detect_res, det_img = bike_detect(test_img)
# 执行违规判断
illegal_res, mark_img = judge_illegal(detect_res, det_img)
# 计算违规车辆数量和合法车辆数量
bike_count = sum(1 for item in detect_res if item[0] == 0)  # 自行车数量
illegal_count = len(illegal_res)  # 违规数量
legal_count = bike_count - illegal_count  # 合法数量
# 保存违规记录
save_status, save_res = save_illegal_record(illegal_res, mark_img, os.path.basename(test_img), "默认场景", illegal_count, legal_count)

# 保存检测结果图片到指定目录
output_path = os.path.join(output_dir, f"detected_{os.path.basename(test_img)}")
cv2.imwrite(output_path, mark_img)

# 打印结果
print(f"检测结果：{detect_res}")
print(f"违规结果：{illegal_res}")
print(f"保存状态：{save_status}, {save_res}")
print(f"检测结果图片已保存到：{output_path}")
