from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame)
from datetime import datetime

class TestPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_path = "F:\\YOLOv10_Bike_Detection\\merged_bicycle_split\\runs\\detect\\bike_yellow_grid_300e\\weights\\best.pt"
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 页面标题
        title_label = QLabel("系统测试")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 30px;
            }
        """)
        layout.addWidget(title_label)
        
        # 测试选项卡片
        test_card = QWidget()
        test_card_layout = QVBoxLayout(test_card)
        test_card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                padding: 25px;
            }
        """)
        
        # 测试选项
        test_options = [
            ("摄像头测试", "测试摄像头连接和实时画面", "#4096ff"),
            ("模型测试", "测试模型性能和准确率", "#722ed1"),
            ("批量测试", "批量测试多张图片", "#67c23a"),
            ("视频测试", "测试视频检测功能", "#fa8c16")
        ]
        
        for name, desc, color in test_options:
            option_widget = QWidget()
            option_layout = QHBoxLayout(option_widget)
            
            info_widget = QWidget()
            info_layout = QVBoxLayout(info_widget)
            
            name_label = QLabel(name)
            name_label.setStyleSheet(f"QLabel {{ font-size: 18px; font-weight: 600; color: {color}; margin-bottom: 5px; }}")
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("QLabel { font-size: 14px; color: #666; }")
            
            info_layout.addWidget(name_label)
            info_layout.addWidget(desc_label)
            
            button = QPushButton("开始测试")
            button.setStyleSheet(f"""
                QPushButton {{
                    padding: 12px 24px;
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 16px;
                }}
                QPushButton:hover {{
                    opacity: 0.9;
                }}
            """)
            
            option_layout.addWidget(info_widget, 1)
            option_layout.addWidget(button)
            
            test_card_layout.addWidget(option_widget)
            
            if name != test_options[-1][0]:
                divider = QFrame()
                divider.setFrameShape(QFrame.HLine)
                divider.setFrameShadow(QFrame.Sunken)
                divider.setStyleSheet("background-color: #f0f2f5; margin: 15px 0;")
                test_card_layout.addWidget(divider)
        
        layout.addWidget(test_card, 2)
        
        # 系统信息
        info_card = QWidget()
        info_card_layout = QVBoxLayout(info_card)
        info_card.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 12px;
                padding: 25px;
            }
        """)
        
        info_title = QLabel("系统信息")
        info_title.setStyleSheet("QLabel { font-size: 18px; font-weight: 600; color: #2c3e50; margin-bottom: 20px; }")
        info_card_layout.addWidget(info_title)
        
        info_text = QLabel(f"""
            模型路径：{self.model_path}
            检测类别：自行车、黄格子
            系统状态：正常运行
            最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)
        info_text.setStyleSheet("QLabel { font-size: 15px; color: #666; line-height: 1.8; }")
        info_card_layout.addWidget(info_text)
        
        layout.addWidget(info_card, 1)
        
        layout.addStretch()
        
        return self
