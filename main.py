# main.py
import os
# 导入所有模块
from detect_model import bike_detect
from illegal_judge import judge_illegal

from record_manage import save_illegal_record, query_record, export_to_excel
from ui_base import BikeDetectionUI
import sys
from PyQt5.QtWidgets import QApplication

# 模型训练函数（后续数据集到位后，添加到这里）
def train_model(data_yaml, epochs=100, batch_size=8, imgsz=640):
    from ultralytics import YOLO
    model = YOLO("yolov10s.pt")
    # 训练模型（改data_yaml为你的数据集配置文件路径）
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        save=True,
        save_dir="models",
        device="0" if os.path.exists("/dev/nvidia0") else "cpu"  # 有显卡用GPU，无则CPU
    )
    # 验证模型
    val_results = model.val()
    return results, val_results

if __name__ == "__main__":
    # 方式1：运行可视化界面（推荐）
    app = QApplication(sys.argv)
    main_window = BikeDetectionUI()
    main_window.show()
    sys.exit(app.exec_())
    # 方式2：单独运行检测+保存（注释上面，打开下面）
    # test_img = "test_bike.jpg"
    # detect_res, det_img = bike_detect(test_img)
    # illegal_res, mark_img = judge_illegal(detect_res, det_img)
    # save_status, save_res = save_illegal_record(illegal_res, mark_img, os.path.basename(test_img))
    # print(f"检测结果：{detect_res}")
    # print(f"违规结果：{illegal_res}")
    # print(f"保存状态：{save_status}, {save_res}")

    # 方式3：训练模型（数据集到位后，打开下面，改data_yaml路径）
    # data_yaml = "data/data.yaml"
    # train_model(data_yaml)