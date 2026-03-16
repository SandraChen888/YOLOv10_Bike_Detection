import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QTextEdit,
    QFileDialog, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO


# ---------------------- 检测线程（功能不变） ----------------------
class DetectionThread(QThread):
    update_frame = pyqtSignal(np.ndarray)  # 传递检测后的帧
    update_log = pyqtSignal(str)  # 传递日志信息
    update_info = pyqtSignal(dict)  # 传递检测信息

    def __init__(self, model_path, source_type, source_path, conf_thres, iou_thres):
        super().__init__()
        self.model = YOLO(model_path)  # 加载YOLOv10模型
        self.source_type = source_type  # image/video/camera
        self.source_path = source_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.is_running = True
        self.current_frame = None  # 保存当前帧用于导出

    def run(self):
        try:
            # 初始化数据源
            if self.source_type == "camera":
                cap = cv2.VideoCapture(0)  # 摄像头
                if not cap.isOpened():
                    self.update_log.emit("❌ 摄像头打开失败！")
                    return
            elif self.source_type == "video":
                cap = cv2.VideoCapture(self.source_path)
                if not cap.isOpened():
                    self.update_log.emit(f"❌ 视频文件打开失败：{self.source_path}")
                    return
            else:  # 单张图片
                img = cv2.imread(self.source_path)
                if img is None:
                    self.update_log.emit(f"❌ 图片文件打开失败：{self.source_path}")
                    return
                self._detect_single_image(img)
                return

            # 视频/摄像头逐帧检测
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.update_log.emit("📹 视频/摄像头流结束")
                    break

                # YOLOv10检测
                results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres)
                det_frame = results[0].plot()  # 绘制检测框
                self.current_frame = det_frame

                # 统计检测信息
                bike_count = len(results[0].boxes)
                infer_time = round(results[0].speed.get("inference", 0), 2)
                conf_list = [round(box.conf.item(), 2) for box in results[0].boxes]

                # 发送信号更新界面
                self.update_frame.emit(det_frame)
                self.update_info.emit({
                    "count": bike_count,
                    "time": infer_time,
                    "conf": conf_list
                })

            cap.release()
        except Exception as e:
            self.update_log.emit(f"❌ 检测出错：{str(e)}")

    def _detect_single_image(self, img):
        """单张图片检测逻辑"""
        results = self.model(img, conf=self.conf_thres, iou=self.iou_thres)
        det_img = results[0].plot()
        self.current_frame = det_img

        bike_count = len(results[0].boxes)
        infer_time = round(results[0].speed.get("inference", 0), 2)
        conf_list = [round(box.conf.item(), 2) for box in results[0].boxes]

        self.update_frame.emit(det_img)
        self.update_info.emit({
            "count": bike_count,
            "time": infer_time,
            "conf": conf_list
        })
        self.update_log.emit(f"✅ 图片检测完成，识别到 {bike_count} 辆自行车")

    def stop(self):
        """停止检测"""
        self.is_running = False


# ---------------------- 主界面类（重点美化样式） ----------------------
class BikeDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv10 自行车检测系统（毕业设计）")
        self.setGeometry(100, 100, 1300, 850)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
                font-family: "Microsoft YaHei", Arial, sans-serif;
            }
            QWidget {
                font-size: 14px;
                color: #333;
            }
            QLabel {
                color: #444;
            }
            QSlider::groove:horizontal {
                border: none;
                height: 8px;
                background: #e5e9f0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4096ff;
                border: none;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #3088ff;
            }
            QComboBox {
                padding: 8px 12px;
                border: 1px solid #dcdfe6;
                border-radius: 6px;
                background-color: white;
                selection-background-color: #e8f4ff;
            }
            QComboBox:hover {
                border-color: #c0c4cc;
            }
            QComboBox::drop-down {
                border: none;
            }
            QTextEdit {
                border: 1px solid #dcdfe6;
                border-radius: 6px;
                padding: 8px;
                background-color: white;
                font-size: 13px;
            }
            QTextEdit:focus {
                border-color: #4096ff;
                outline: none;
            }
        """)

        self.detection_thread = None
        self.model_path = "yolov10s.pt"  # 替换为你的模型路径
        self.source_path = ""  # 数据源路径

        # 初始化界面
        self._init_ui()

    def _init_ui(self):
        """初始化界面布局（美化版）"""
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ========== 左侧功能区（卡片式设计） ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        # 左侧卡片样式
        left_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            }
        """)
        left_widget.setMaximumWidth(350)
        main_layout.addWidget(left_widget)

        # 1. 标题栏
        title_label = QLabel("YOLOv10 检测控制台")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 25px;
                padding-bottom: 10px;
                border-bottom: 1px solid #f0f2f5;
            }
        """)
        left_layout.addWidget(title_label)

        # 2. 数据源选择
        source_label = QLabel("📁 数据源选择")
        source_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        """)
        left_layout.addWidget(source_label)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["本地图片", "本地视频", "摄像头实时"])
        left_layout.addWidget(self.source_combo)
        left_layout.addSpacing(8)

        self.select_source_btn = QPushButton("选择文件")
        self.select_source_btn.clicked.connect(self._select_source)
        self.select_source_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3088ff;
            }
            QPushButton:pressed {
                background-color: #2078ee;
            }
        """)
        left_layout.addWidget(self.select_source_btn)
        left_layout.addSpacing(25)

        # 3. 检测参数调节
        param_label = QLabel("⚙️ 检测参数")
        param_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        """)
        left_layout.addWidget(param_label)

        # 置信度滑块
        conf_layout = QHBoxLayout()
        conf_layout.setSpacing(10)
        conf_label = QLabel("置信度：")
        conf_label.setStyleSheet("color: #666;")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)  # 默认0.5
        self.conf_label = QLabel("0.5")
        self.conf_label.setStyleSheet("""
            QLabel {
                color: #4096ff;
                font-weight: 600;
                min-width: 40px;
                text-align: center;
            }
        """)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(str(v / 100)))
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        left_layout.addLayout(conf_layout)
        left_layout.addSpacing(10)

        # IoU滑块
        iou_layout = QHBoxLayout()
        iou_layout.setSpacing(10)
        iou_label = QLabel("IoU阈值：")
        iou_label.setStyleSheet("color: #666;")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)  # 默认0.45
        self.iou_label = QLabel("0.45")
        self.iou_label.setStyleSheet("""
            QLabel {
                color: #4096ff;
                font-weight: 600;
                min-width: 40px;
                text-align: center;
            }
        """)
        self.iou_slider.valueChanged.connect(lambda v: self.iou_label.setText(str(v / 100)))
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_label)
        left_layout.addLayout(iou_layout)
        left_layout.addSpacing(25)

        # 4. 操作按钮（横向排列）
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.start_btn = QPushButton("▶️ 开始检测")
        self.start_btn.clicked.connect(self._start_detection)
        self.start_btn.setStyleSheet("""
            QPushButton {
                padding: 12px;
                background-color: #67c23a;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                flex: 1;
            }
            QPushButton:hover {
                background-color: #5daf34;
            }
            QPushButton:pressed {
                background-color: #529e2d;
            }
            QPushButton:disabled {
                background-color: #b3e19d;
                color: #f5f5f5;
            }
        """)

        self.stop_btn = QPushButton("⏹️ 停止检测")
        self.stop_btn.clicked.connect(self._stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                padding: 12px;
                background-color: #f56c6c;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                flex: 1;
            }
            QPushButton:hover {
                background-color: #e45858;
            }
            QPushButton:pressed {
                background-color: #d34848;
            }
            QPushButton:disabled {
                background-color: #f9b4b4;
                color: #f5f5f5;
            }
        """)

        self.export_btn = QPushButton("💾 导出结果")
        self.export_btn.clicked.connect(self._export_result)
        self.export_btn.setStyleSheet("""
            QPushButton {
                padding: 12px;
                background-color: #f7ba1e;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                flex: 1;
            }
            QPushButton:hover {
                background-color: #e6ac1c;
            }
            QPushButton:pressed {
                background-color: #d59e1a;
            }
        """)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.export_btn)
        left_layout.addLayout(btn_layout)
        left_layout.addSpacing(25)

        # 5. 模型信息（卡片式）
        model_card = QWidget()
        model_card_layout = QVBoxLayout(model_card)
        model_card.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
            }
        """)

        model_label = QLabel("📊 模型信息")
        model_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
            }
        """)
        model_card_layout.addWidget(model_label)

        self.model_info_label = QLabel(f"""当前模型：{self.model_path}
检测类别：自行车
运行设备：CPU/GPU（自动适配）""")
        self.model_info_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #666;
                line-height: 1.5;
            }
        """)
        model_card_layout.addWidget(self.model_info_label)
        left_layout.addWidget(model_card)

        # 填充空白
        left_layout.addStretch()

        # ========== 右侧结果展示区（卡片式） ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            }
        """)
        main_layout.addWidget(right_widget, 3)

        # 1. 检测结果预览
        result_title_layout = QHBoxLayout()
        result_label = QLabel("检测结果预览")
        result_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #2c3e50;
            }
        """)
        result_title_layout.addWidget(result_label)
        result_title_layout.addStretch()
        right_layout.addLayout(result_title_layout)
        right_layout.addSpacing(10)

        self.result_display = QLabel()
        self.result_display.setStyleSheet("""
            QLabel {
                border: 1px solid #e5e9f0;
                border-radius: 8px;
                background-color: #f8f9fa;
                color: #909399;
                font-size: 15px;
            }
        """)
        self.result_display.setAlignment(Qt.AlignCenter)
        self.result_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 默认显示提示图
        self.result_display.setText("请选择数据源并点击开始检测")
        right_layout.addWidget(self.result_display, 6)
        right_layout.addSpacing(15)

        # 2. 检测信息面板
        info_label = QLabel("📋 检测信息")
        info_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
            }
        """)
        right_layout.addWidget(info_label)

        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setText("自行车数量：0 辆\n单帧检测耗时：0 ms\n置信度分布：[]")
        right_layout.addWidget(self.info_panel, 1)
        right_layout.addSpacing(10)

        # 3. 日志输出
        log_label = QLabel("📝 系统日志")
        log_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
            }
        """)
        right_layout.addWidget(log_label)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setText("系统已就绪，等待检测...")
        right_layout.addWidget(self.log_display, 1)

    def _select_source(self):
        """选择图片/视频文件"""
        source_type = self.source_combo.currentText()
        if source_type == "本地图片":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
        elif source_type == "本地视频":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
            )
        else:  # 摄像头
            file_path = "camera"
            self.log_display.append("✅ 已选择摄像头作为数据源")
            self.source_path = file_path
            return

        if file_path:
            self.source_path = file_path
            self.log_display.append(f"✅ 已选择数据源：{file_path}")
        else:
            self.log_display.append("❌ 未选择任何文件")

    def _start_detection(self):
        """启动检测"""
        # 检查数据源
        if self.source_combo.currentText() != "摄像头实时" and not self.source_path:
            self.log_display.append("❌ 请先选择图片/视频文件！")
            return

        # 获取参数
        conf_thres = self.conf_slider.value() / 100
        iou_thres = self.iou_slider.value() / 100
        source_type_map = {"本地图片": "image", "本地视频": "video", "摄像头实时": "camera"}
        source_type = source_type_map[self.source_combo.currentText()]

        # 启动检测线程
        self.detection_thread = DetectionThread(
            model_path=self.model_path,
            source_type=source_type,
            source_path=self.source_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )

        # 绑定信号
        self.detection_thread.update_frame.connect(self._update_frame)
        self.detection_thread.update_log.connect(self._update_log)
        self.detection_thread.update_info.connect(self._update_info)
        self.detection_thread.finished.connect(self._reset_btn_state)

        # 启动线程
        self.detection_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_display.append("✅ 开始检测...")

    def _stop_detection(self):
        """停止检测"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.log_display.append("⏹️ 已停止检测")

    def _reset_btn_state(self):
        """重置按钮状态"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _update_frame(self, frame):
        """更新检测帧到界面"""
        # 转换OpenCV帧为Qt格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放适配显示区域
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.result_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.result_display.setPixmap(pixmap)

    def _update_info(self, info):
        """更新检测信息面板"""
        info_text = f"""
自行车数量：{info['count']} 辆
单帧检测耗时：{info['time']} ms
置信度分布：{info['conf']}
        """
        self.info_panel.setText(info_text.strip())

    def _update_log(self, msg):
        """更新日志"""
        self.log_display.append(msg)

    def _export_result(self):
        """导出检测结果"""
        if not hasattr(self.detection_thread, "current_frame") or self.detection_thread.current_frame is None:
            self.log_display.append("❌ 暂无检测结果可导出！")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", "", "Images (*.png *.jpg *.jpeg)"
        )
        if save_path:
            cv2.imwrite(save_path, self.detection_thread.current_frame)
            self.log_display.append(f"✅ 结果已导出至：{save_path}")


# 测试入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    window = BikeDetectionUI()
    window.show()
    sys.exit(app.exec_())