from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                           QComboBox, QSlider, QTextEdit, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt

class DetectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_path = "F:\\YOLOv10_Bike_Detection\\merged_bicycle_split\\runs\\detect\\bike_yellow_grid_300e\\weights\\best.pt"
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
        left_widget.setMaximumWidth(400)
        main_layout.addWidget(left_widget, 1)

        # 1. 标题栏
        title_label = QLabel("操作空间")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 22px;
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
                font-size: 18px;
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
        
        left_layout.addLayout(upload_btn_layout)
        left_layout.addSpacing(30)

        # 3. 场景选择
        scene_label = QLabel("场景选择")
        scene_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
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
                font-size: 17px;
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
                font-size: 18px;
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
        conf_label.setStyleSheet("color: #666; font-size: 17px;")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)  # 默认0.5
        self.conf_label = QLabel("0.5")
        self.conf_label.setStyleSheet("""
            QLabel {
                color: #4096ff;
                font-weight: 600;
                font-size: 17px;
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
        iou_label.setStyleSheet("color: #666; font-size: 17px;")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)  # 默认0.45
        self.iou_label = QLabel("0.45")
        self.iou_label.setStyleSheet("""
            QLabel {
                color: #4096ff;
                font-weight: 600;
                font-size: 17px;
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
                font-size: 15px;
            }
        """)
        self.source_combo.hide()  # 隐藏，通过按钮控制
        left_layout.addWidget(self.source_combo)
        left_layout.addSpacing(10)

        # 摄像头索引选择
        self.camera_index_widget = QWidget()
        self.camera_index_layout = QHBoxLayout(self.camera_index_widget)
        camera_index_label = QLabel("摄像头索引：")
        camera_index_label.setStyleSheet("color: #666; font-size: 15px;")
        self.camera_index_combo = QComboBox()
        self.camera_index_combo.addItems(["0 (默认)", "1", "2", "3"])
        self.camera_index_combo.setStyleSheet("""
            QComboBox {
                padding: 12px 16px;
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #e8f4ff;
                font-size: 15px;
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
                font-size: 18px;
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
                font-size: 20px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 15px;
            }
        """)
        right_layout.addWidget(visual_label)

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
                font-size: 14px;
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
                font-size: 17px;
            }
        """)
        self.result_display.setAlignment(Qt.AlignCenter)
        self.result_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 默认显示提示图
        self.result_display.setText("请上传文件并点击开始检测")
        right_layout.addWidget(self.result_display, 5)
        right_layout.addSpacing(20)

        # 2. 检测结果统计区
        stats_label = QLabel("检测结果统计")
        stats_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
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
                    font-size: 16px;
                    color: #666;
                    margin-bottom: 8px;
                }
            """)
            
            stat_value = QLabel(value)
            stat_value.setStyleSheet(f"QLabel {{ font-size: 24px; font-weight: bold; color: {color}; }}")
            
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
                font-size: 16px;
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
                font-size: 14px;
                color: #2c3e50;
                padding: 15px;
            }
        """)
        right_layout.addWidget(self.log_display, 2)
        
        return self
