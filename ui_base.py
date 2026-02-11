# ui_base.py
import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QFileDialog, QVBoxLayout, QWidget, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
# 导入自己写的模块
from detect_model import bike_detect
from illegal_judge import judge_illegal
from record_manage import save_illegal_record


class BikeDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None  # 选中的图片路径

    def initUI(self):
        # 窗口设置
        self.setWindowTitle("校园单车违规停放检测系统")
        self.setGeometry(100, 100, 1000, 700)

        # 布局
        layout = QVBoxLayout()

        # 按钮：选择图片
        self.btn_select = QPushButton("选择检测图片")
        self.btn_select.clicked.connect(self.select_image)
        layout.addWidget(self.btn_select)

        # 按钮：开始检测
        self.btn_detect = QPushButton("开始违规检测")
        self.btn_detect.clicked.connect(self.start_detect)
        self.btn_detect.setEnabled(False)
        layout.addWidget(self.btn_detect)

        # 图片展示标签
        self.label_img = QLabel("请选择图片")
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setStyleSheet("border: 1px solid #ccc;")
        layout.addWidget(self.label_img)

        # 结果展示文本框
        self.text_res = QTextEdit()
        self.text_res.setPlaceholderText("检测结果将显示在这里...")
        self.text_res.setReadOnly(True)
        layout.addWidget(self.text_res)

        # 中心窗口
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def select_image(self):
        # 打开文件对话框，选择图片
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            self.image_path = file_path
            # 显示选中的图片
            self.show_image(file_path)
            self.btn_detect.setEnabled(True)
            self.text_res.setText(f"已选择图片：{os.path.basename(file_path)}")

    def show_image(self, img_path, img=None):
        # 展示图片（支持本地路径或cv2图片）
        if img is None:
            img = cv2.imread(img_path)
        # 转换cv2图片为Qt格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 缩放图片适配窗口
        pixmap = QPixmap.fromImage(q_img).scaled(self.label_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_img.setPixmap(pixmap)

    def start_detect(self):
        if not self.image_path:
            self.text_res.setText("请先选择图片！")
            return
        try:
            # 1. 目标检测
            detect_res, det_img = bike_detect(self.image_path)
            if not detect_res:
                self.text_res.setText("未检测到单车！")
                self.show_image(None, det_img)
                return
            # 2. 违规判定
            illegal_res, mark_img = judge_illegal(detect_res, det_img)
            # 3. 保存记录
            image_name = os.path.basename(self.image_path)
            save_status, save_res = save_illegal_record(illegal_res, mark_img, image_name)
            # 4. 展示结果
            self.show_image(None, mark_img)
            # 整理结果文本
            res_text = f"检测完成！共检测到{len(detect_res)}辆单车，其中违规{len(illegal_res)}辆\n"
            res_text += f"记录保存状态：{'成功，记录ID：' + save_res if save_status else '失败：' + save_res}\n"
            for i, res in enumerate(illegal_res):
                res_text += f"【违规单车{i + 1}】类型：{res['违规类型']}，置信度：{res['置信度']:.2f}，重叠占比：{res['重叠占比']:.2f}\n"
            self.text_res.setText(res_text)
        except Exception as e:
            self.text_res.setText(f"检测失败：{str(e)}")


# 运行界面
if __name__ == "__main__":
    # 安装PyQt5（如果未安装）：pip install pyqt5
    app = QApplication(sys.argv)
    window = BikeDetectionUI()
    window.show()
    sys.exit(app.exec_())