from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                           QComboBox, QSlider, QTextEdit, QFrame, QSizePolicy, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
from detection_thread import DetectionThread

class DetectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_path = "F:\YOLOv10_Bike_Detection\merged_bicycle_split\runs\detect\bike_yellow_grid_300e\weights\best.pt"
        # 初始化属性
        self.source_path = None
        self.batch_results = []
        self.current_batch_index = 0
        self.detection_thread = None
        # 用于保存连续上传的单张图片结果
        self.single_image_results = []
        self.current_single_image_index = 0
        self._init_ui()
    
    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ========== 左侧操作空间区（卡片式设计） ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        # 左侧卡片样式
        left_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        # 调整左侧操作空间的最大宽度，使其在应用放大时保持适当的比例
        left_widget.setMaximumWidth(450)
        # 调整左侧操作空间的最小宽度，确保在小窗口时也能正常显示
        left_widget.setMinimumWidth(350)
        main_layout.addWidget(left_widget, 1)

        # 1. 标题栏
        title_label = QLabel("操作空间")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 30px;
                padding-bottom: 12px;
                border-bottom: 1px solid #f0f2f5;
            }
        """)
        left_layout.addWidget(title_label)

        # 2. 文件上传区域
        upload_label = QLabel("文件上传")
        upload_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
            }
        """)
        left_layout.addWidget(upload_label)

        # 上传按钮组
        upload_btn_layout = QVBoxLayout()
        upload_btn_layout.setSpacing(12)
        
        self.upload_image_btn = QPushButton("上传图片")
        self.upload_image_btn.setStyleSheet("""
            QPushButton {
                padding: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4096ff, stop:1 #3088ff);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 17px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #50a6ff, stop:1 #4096ff);
            }
        """)
        upload_btn_layout.addWidget(self.upload_image_btn)
        
        self.upload_video_btn = QPushButton("上传视频")
        self.upload_video_btn.setStyleSheet("""
            QPushButton {
                padding: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #722ed1, stop:1 #5217b8);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 17px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #853fe1, stop:1 #6222c4);
            }
        """)
        upload_btn_layout.addWidget(self.upload_video_btn)
        
        self.upload_camera_btn = QPushButton("摄像头实时")
        self.upload_camera_btn.setStyleSheet("""
            QPushButton {
                padding: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #67c23a, stop:1 #529e2d);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 17px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #7dd54a, stop:1 #5daf34);
            }
        """)
        upload_btn_layout.addWidget(self.upload_camera_btn)
        
        self.upload_batch_btn = QPushButton("批量上传")
        self.upload_batch_btn.setStyleSheet("""
            QPushButton {
                padding: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #faad14, stop:1 #fa8c16);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 17px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #fab934, stop:1 #fb9a36);
            }
        """)
        upload_btn_layout.addWidget(self.upload_batch_btn)
        
        left_layout.addLayout(upload_btn_layout)
        left_layout.addSpacing(30)

        # 3. 场景选择
        scene_label = QLabel("场景选择")
        scene_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
            }
        """)
        left_layout.addWidget(scene_label)

        self.scene_combo = QComboBox()
        self.scene_combo.addItems([
            "教学楼A栋",
            "教学楼B栋",
            "教学楼C栋",
            "教学楼D栋",
            "教学楼E栋",
            "教学楼F栋",
            "一饭堂门口"
        ])
        self.scene_combo.setStyleSheet("""
            QComboBox {
                padding: 14px 18px;
                border: 2px solid #e5e9f0;
                border-radius: 10px;
                background-color: white;
                selection-background-color: #e8f4ff;
                font-size: 19px;
            }
            QComboBox:hover {
                border-color: #c6e2ff;
            }
            QComboBox:focus {
                border-color: #4096ff;
            }
        """)
        left_layout.addWidget(self.scene_combo)
        left_layout.addSpacing(30)

        # 4. 检测参数
        param_label = QLabel("检测参数")
        param_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
            }
        """)
        left_layout.addWidget(param_label)

        # 置信度滑块
        conf_layout = QHBoxLayout()
        conf_layout.setSpacing(12)
        conf_label = QLabel("置信度：")
        conf_label.setStyleSheet("color: #666; font-size: 19px;")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)  # 默认0.5
        self.conf_slider.setMinimumWidth(100)  # 设置最小宽度
        self.conf_slider.setStyleSheet("""
            QSlider {
                height: 8px;
                min-width: 100px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #dcdfe6;
                height: 6px;
                background: #f0f2f5;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4096ff;
                border: 2px solid white;
                width: 20px;
                height: 20px;
                border-radius: 10px;
                margin: -7px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #66b1ff;
            }
            QSlider::handle:horizontal:pressed {
                background: #3385ff;
            }
        """)
        self.conf_label = QLabel("0.5")
        self.conf_label.setStyleSheet("""
            QLabel {
                color: #4096ff;
                font-weight: 600;
                font-size: 19px;
                min-width: 50px;
                text-align: center;
            }
        """)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(str(v / 100)))
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        left_layout.addLayout(conf_layout)
        left_layout.addSpacing(12)

        # IOU阈值滑块
        iou_layout = QHBoxLayout()
        iou_layout.setSpacing(12)
        iou_label = QLabel("IOU阈值：")
        iou_label.setStyleSheet("color: #666; font-size: 19px;")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)  # 默认0.45
        self.iou_slider.setMinimumWidth(100)  # 设置最小宽度
        self.iou_slider.setStyleSheet("""
            QSlider {
                height: 8px;
                min-width: 100px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #dcdfe6;
                height: 6px;
                background: #f0f2f5;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4096ff;
                border: 2px solid white;
                width: 20px;
                height: 20px;
                border-radius: 10px;
                margin: -7px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #66b1ff;
            }
            QSlider::handle:horizontal:pressed {
                background: #3385ff;
            }
        """)
        self.iou_label = QLabel("0.45")
        self.iou_label.setStyleSheet("""
            QLabel {
                color: #4096ff;
                font-weight: 600;
                font-size: 19px;
                min-width: 50px;
                text-align: center;
            }
        """)
        self.iou_slider.valueChanged.connect(lambda v: self.iou_label.setText(str(v / 100)))
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_label)
        left_layout.addLayout(iou_layout)
        left_layout.addSpacing(12)



        # 数据源选择（隐藏在操作区）
        self.source_combo = QComboBox()
        self.source_combo.addItems(["本地图片", "批量图片", "本地视频", "摄像头实时"])
        self.source_combo.setStyleSheet("""
            QComboBox {
                padding: 12px 16px;
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #e8f4ff;
                font-size: 17px;
            }
        """)
        self.source_combo.hide()  # 隐藏，通过按钮控制
        left_layout.addWidget(self.source_combo)
        left_layout.addSpacing(10)

        # 摄像头索引选择
        self.camera_index_widget = QWidget()
        self.camera_index_layout = QHBoxLayout(self.camera_index_widget)
        camera_index_label = QLabel("摄像头索引：")
        camera_index_label.setStyleSheet("color: #666; font-size: 17px;")
        self.camera_index_combo = QComboBox()
        self.camera_index_combo.addItems(["0 (默认)", "1", "2", "3"])
        self.camera_index_combo.setStyleSheet("""
            QComboBox {
                padding: 12px 16px;
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #e8f4ff;
                font-size: 17px;
            }
        """)
        self.camera_index_layout.addWidget(camera_index_label)
        self.camera_index_layout.addWidget(self.camera_index_combo)
        self.camera_index_widget.hide()  # 隐藏，需要时显示
        left_layout.addWidget(self.camera_index_widget)
        left_layout.addSpacing(10)

        # 选择文件按钮（隐藏，通过上传按钮触发）
        self.select_source_btn = QPushButton("选择文件")
        self.select_source_btn.setStyleSheet("""
            QPushButton {
                padding: 12px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 15px;
            }
        """)
        self.select_source_btn.hide()  # 隐藏，通过上传按钮触发
        left_layout.addWidget(self.select_source_btn)
        left_layout.addSpacing(30)

        # 5. 操作按钮
        action_label = QLabel("操作控制")
        action_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
            }
        """)
        left_layout.addWidget(action_label)

        # 创建统一的按钮样式
        button_style = """
            QPushButton {
                padding: 16px 24px;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 17px;
            }
            QPushButton:hover {
                transform: translateY(-2px);
            }
        """

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setStyleSheet("""
            QPushButton {
                padding: 18px 24px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #67c23a, stop:1 #529e2d);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #7dd54a, stop:1 #5daf34);
            }
            QPushButton:disabled {
                background: #b3e19d;
                color: #f5f5f5;
            }
        """)

        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                padding: 18px 24px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f56c6c, stop:1 #d34848);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff7d7d, stop:1 #e45858);
            }
            QPushButton:disabled {
                background: #f9b4b4;
                color: #f5f5f5;
            }
        """)

        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        left_layout.addLayout(btn_layout)
        left_layout.addSpacing(25)

        # 6. 模型信息
        model_card = QWidget()
        model_card_layout = QVBoxLayout(model_card)
        model_card.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 16px;
            }
        """)

        model_label = QLabel("模型信息")
        model_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
            }
        """)
        model_card_layout.addWidget(model_label)

        self.model_info_label = QLabel(f"""当前模型：{self.model_path}
检测类别：自行车、黄格子
运行设备：CPU/GPU（自动适配）""")
        self.model_info_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
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
            }
        """)
        main_layout.addWidget(right_widget, 3)

        # 1. 核心可视化展示区域
        visual_label = QLabel("核心可视化展示")
        visual_label.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 15px;
            }
        """)
        right_layout.addWidget(visual_label)

        # FPS 显示器
        self.fps_display = QLabel("FPS: 0.00")
        self.fps_display.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #4096ff;
                background-color: #f8f9fa;
                padding: 8px 16px;
                border-radius: 8px;
                margin-bottom: 10px;
                text-align: center;
            }
        """)
        right_layout.addWidget(self.fps_display)

        # 预览图片导航按钮
        self.nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.setEnabled(False)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3088ff;
            }
            QPushButton:disabled {
                background-color: #c6e2ff;
                color: #a0cfff;
            }
        """)
        self.next_btn = QPushButton("下一张")
        self.next_btn.setEnabled(False)
        self.next_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3088ff;
            }
            QPushButton:disabled {
                background-color: #c6e2ff;
                color: #a0cfff;
            }
        """)
        self.image_index_label = QLabel("0/0")
        self.image_index_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #666;
                margin: 0 10px;
            }
        """)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.prev_btn)
        self.nav_layout.addWidget(self.image_index_label)
        self.nav_layout.addWidget(self.next_btn)
        right_layout.addLayout(self.nav_layout)
        right_layout.addSpacing(12)

        self.result_display = QLabel()
        self.result_display.setStyleSheet("""
            QLabel {
                border: 2px solid #e5e9f0;
                border-radius: 12px;
                background-color: #f8f9fa;
                color: #909399;
                font-size: 19px;
            }
        """)
        self.result_display.setAlignment(Qt.AlignCenter)
        # 设置固定的最小大小，防止区域大小变化
        self.result_display.setMinimumSize(800, 500)
        # 设置最大高度，防止窗口放大时过度扩展
        self.result_display.setMaximumHeight(800)
        # 设置大小策略，使其在应用放大时能够适当扩展，但不挤压下方区域
        self.result_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # 默认显示提示图
        self.result_display.setText("请上传文件并点击开始检测")
        # 调整可视化展示区域的比例，使其在应用放大时更加美观
        right_layout.addWidget(self.result_display, 4)
        right_layout.addSpacing(20)

        # 2. 检测结果统计区
        stats_label = QLabel("检测结果统计")
        stats_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
            }
        """)
        right_layout.addWidget(stats_label)

        # 统计信息卡片
        stats_card = QWidget()
        stats_card_layout = QGridLayout(stats_card)
        stats_card.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 12px;
                padding: 25px;
            }
        """)

        # 统计项
        stats_items = [
            ("检测耗时", "0 ms", "#4096ff"),
            ("违规车辆数量", "0 辆", "#f56c6c"),
            ("合法车辆数量", "0 辆", "#67c23a"),
            ("总检测车辆", "0 辆", "#fa8c16")
        ]

        # 保存统计值标签的引用
        self.stats_labels = {}
        
        for i, (label, value, color) in enumerate(stats_items):
            stat_label = QLabel(label)
            stat_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    color: #666;
                    margin-bottom: 8px;
                }
            """)
            
            stat_value = QLabel(value)
            stat_value.setStyleSheet(f"QLabel {{ font-size: 26px; font-weight: bold; color: {color}; }}")
            
            stats_card_layout.addWidget(stat_label, i, 0)
            stats_card_layout.addWidget(stat_value, i, 1)
            
            # 保存引用
            self.stats_labels[label] = stat_value
            
            if i < len(stats_items) - 1:
                divider = QFrame()
                divider.setFrameShape(QFrame.HLine)
                divider.setFrameShadow(QFrame.Sunken)
                divider.setStyleSheet("background-color: #e5e9f0;")
                stats_card_layout.addWidget(divider, i+1, 0, 1, 2)
                stats_card_layout.setRowMinimumHeight(i+1, 15)

        right_layout.addWidget(stats_card)
        right_layout.addSpacing(15)

        # 3. 系统日志
        log_label = QLabel("系统日志")
        log_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        """)
        right_layout.addWidget(log_label)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setText("系统已就绪，等待检测...")
        self.log_display.setStyleSheet("""
            QTextEdit {
                border: 2px solid #e5e9f0;
                border-radius: 12px;
                background-color: #f8f9fa;
                padding: 15px;
                font-size: 16px;
                color: #333;
            }
        """)
        # 调整系统日志区的比例，使其在应用放大时更加美观
        right_layout.addWidget(self.log_display, 2)
        
        # 连接按钮点击事件
        self.upload_image_btn.clicked.connect(lambda: self._select_source_type("image"))
        self.upload_video_btn.clicked.connect(lambda: self._select_source_type("video"))
        self.upload_camera_btn.clicked.connect(lambda: self._select_source_type("camera"))
        self.upload_batch_btn.clicked.connect(lambda: self._select_source_type("batch"))
        self.start_btn.clicked.connect(self._start_detection)
        self.stop_btn.clicked.connect(self._stop_detection)
        self.prev_btn.clicked.connect(self._prev_batch_image)
        self.next_btn.clicked.connect(self._next_batch_image)
        
        return self
    
    def _select_source_type(self, source_type):
        """根据来源类型选择文件"""
        if source_type == "image":
            self.source_combo.setCurrentIndex(0)  # 本地图片
            self._select_source()
            # 隐藏摄像头索引选择
            self.camera_index_widget.hide()
        elif source_type == "video":
            self.source_combo.setCurrentIndex(2)  # 本地视频
            self._select_source()
            # 隐藏摄像头索引选择
            self.camera_index_widget.hide()
        elif source_type == "batch":
            self.source_combo.setCurrentIndex(1)  # 批量图片
            self._select_source()
            # 隐藏摄像头索引选择
            self.camera_index_widget.hide()
        elif source_type == "camera":
            self.source_combo.setCurrentIndex(3)  # 摄像头实时
            self.source_path = "camera"
            # 显示摄像头索引选择
            self.camera_index_widget.show()
            self.log_display.append("✅ 已选择摄像头作为数据源")
            self.log_display.append("📷 请选择正确的摄像头索引")
    
    def _select_source(self):
        """选择图片/视频文件"""
        source_type = self.source_combo.currentText()
        if source_type == "本地图片":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
        elif source_type == "批量图片":
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "选择多张图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
            if file_paths:
                self.source_path = file_paths
                self.log_display.append(f"✅ 已选择 {len(file_paths)} 张图片")
            else:
                self.log_display.append("❌ 未选择任何文件")
            return
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
        source_type_map = {"本地图片": "image", "批量图片": "batch_image", "本地视频": "video", "摄像头实时": "camera"}
        source_type = source_type_map[self.source_combo.currentText()]

        # 获取摄像头索引
        camera_index = 0
        if source_type == "camera":
            # 从下拉框获取摄像头索引
            camera_index_text = self.camera_index_combo.currentText()
            camera_index = int(camera_index_text.split()[0])

        # 获取当前场景
        current_scene = self.scene_combo.currentText()

        # 使用默认重叠阈值0.65
        overlap_thres = 0.65

        # 启动检测线程
        self.detection_thread = DetectionThread(
            model_path=self.model_path,
            source_type=source_type,
            source_path=self.source_path if self.source_path else "",
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            overlap_thresh=overlap_thres,
            camera_index=camera_index,
            scene=current_scene
        )

        # 绑定信号
        self.detection_thread.update_frame.connect(self._update_frame)
        self.detection_thread.update_log.connect(self._update_log)
        self.detection_thread.update_info.connect(self._update_info)
        self.detection_thread.show_video_duration_warning.connect(self._show_video_duration_warning)
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
        
        # 批量检测完成后更新结果
        if self.detection_thread and self.detection_thread.source_type == "batch_image":
            self.batch_results = self.detection_thread.batch_results
            if self.batch_results:
                self.current_batch_index = 0
                self._display_batch_image()
                self.log_display.append(f"✅ 批量检测结果已加载，共 {len(self.batch_results)} 张图片")
        
        # 单张图片检测完成后保存结果
        elif self.detection_thread and self.detection_thread.source_type == "image":
            # 将单张图片的结果添加到 batch_results 中
            # 这样就可以使用现有的导航功能来浏览连续上传的单张图片
            if self.detection_thread.batch_results:
                # 获取最后一个结果（即当前单张图片的结果）
                last_result = self.detection_thread.batch_results[-1]
                # 添加到 batch_results 中
                self.batch_results.append(last_result)
                # 更新当前索引到最后一个
                self.current_batch_index = len(self.batch_results) - 1
                # 显示当前图片
                self._display_batch_image()
                # 启用导航按钮
                self.prev_btn.setEnabled(len(self.batch_results) > 1)
                self.next_btn.setEnabled(len(self.batch_results) > 1)
                # 更新索引标签
                self.image_index_label.setText(f"{self.current_batch_index + 1}/{len(self.batch_results)}")
                # 显示日志
                self.log_display.append(f"✅ 单张图片检测完成，已添加到结果列表，共 {len(self.batch_results)} 张图片")

    def _prev_batch_image(self):
        """显示上一张批量检测图片"""
        if self.batch_results and self.current_batch_index > 0:
            self.current_batch_index -= 1
            self._display_batch_image()

    def _next_batch_image(self):
        """显示下一张批量检测图片"""
        if self.batch_results and self.current_batch_index < len(self.batch_results) - 1:
            self.current_batch_index += 1
            self._display_batch_image()

    def _display_batch_image(self):
        """显示当前批量检测图片"""
        if not self.batch_results:
            return
        
        # 获取当前图片的结果
        result = self.batch_results[self.current_batch_index]
        
        # 更新索引标签
        self.image_index_label.setText(f"{self.current_batch_index + 1}/{len(self.batch_results)}")
        
        # 更新导航按钮状态
        self.prev_btn.setEnabled(self.current_batch_index > 0)
        self.next_btn.setEnabled(self.current_batch_index < len(self.batch_results) - 1)
        
        # 显示当前图片
        frame = result['detected_image']
        
        # 转换为 QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 显示图片
        pixmap = QPixmap.fromImage(qt_img)
        self.result_display.setPixmap(pixmap.scaled(
            self.result_display.width(), self.result_display.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # 更新统计信息
        self.stats_labels["检测耗时"].setText(f"{result['time']} ms")
        self.stats_labels["违规车辆数量"].setText(f"{result['illegal_count']} 辆")
        self.stats_labels["合法车辆数量"].setText(f"{result['legal_count']} 辆")
        self.stats_labels["总检测车辆"].setText(f"{result['total_count']} 辆")
        # 更新 FPS 显示
        if 'fps' in result:
            self.fps_display.setText(f"FPS: {result['fps']:.2f}")

    def _update_frame(self, frame):
        """更新帧"""
        # 转换为 QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 显示图片
        pixmap = QPixmap.fromImage(qt_img)
        self.result_display.setPixmap(pixmap.scaled(
            self.result_display.width(), self.result_display.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _update_log(self, message):
        """更新日志"""
        self.log_display.append(message)

    def _update_info(self, info):
        """更新信息"""
        # 更新统计信息
        if 'time' in info:
            self.stats_labels["检测耗时"].setText(f"{info['time']:.2f} ms")
        if 'total' in info:
            self.stats_labels["总检测车辆"].setText(f"{info['total']} 辆")
        if 'illegal' in info:
            self.stats_labels["违规车辆数量"].setText(f"{info['illegal']} 辆")
        if 'legal' in info:
            self.stats_labels["合法车辆数量"].setText(f"{info['legal']} 辆")
        if 'fps' in info:
            self.fps_display.setText(f"FPS: {info['fps']:.2f}")

    def _show_video_duration_warning(self, duration):
        """显示视频时长警告"""
        QMessageBox.warning(
            self, "视频时长提示",
            f"视频时长 {duration:.1f} 秒，超过5分钟限制，建议选择更短的视频进行检测。"
        )
