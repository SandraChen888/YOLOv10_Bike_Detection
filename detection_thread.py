#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测线程模块
"""

import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal
from detect_model import bike_detect
from illegal_judge import judge_illegal
from record_manage import save_illegal_record

class DetectionThread(QThread):
    """检测线程类"""
    
    # 信号定义
    update_frame = pyqtSignal(object)  # 更新帧信号
    update_log = pyqtSignal(str)  # 更新日志信号
    update_info = pyqtSignal(dict)  # 更新信息信号
    show_video_duration_warning = pyqtSignal()  # 视频时长警告信号
    
    def __init__(self, model_path, source_type, source_path, conf_thres, iou_thres, camera_index=0, scene=""):
        super().__init__()
        self.model_path = model_path
        self.source_type = source_type
        self.source_path = source_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.camera_index = camera_index
        self.scene = scene
        self.running = False
    
    def run(self):
        """线程运行方法"""
        self.running = True
        
        try:
            if self.source_type == "image":
                self._process_image()
            elif self.source_type == "batch_image":
                self._process_batch_images()
            elif self.source_type == "video":
                self._process_video()
            elif self.source_type == "camera":
                self._process_camera()
        except Exception as e:
            self.update_log.emit(f"❌ 检测出错：{str(e)}")
        finally:
            self.running = False
    
    def stop(self):
        """停止线程"""
        self.running = False
    
    def _process_image(self):
        """处理单张图片"""
        start_time = time.time()
        
        # 执行检测
        detect_res, det_img = bike_detect(self.source_path, conf=self.conf_thres)
        
        # 执行违规判断
        illegal_res, mark_img = judge_illegal(detect_res, det_img)
        
        # 保存违规记录
        save_status, save_res = save_illegal_record(illegal_res, mark_img, self.source_path, self.scene)
        
        # 计算耗时
        elapsed_time = int((time.time() - start_time) * 1000)
        
        # 统计信息
        bike_count = sum(1 for item in detect_res if item[0] == 0)  # 自行车数量
        illegal_count = len(illegal_res)  # 违规数量
        legal_count = bike_count - illegal_count  # 合法数量
        
        # 发送信号
        self.update_frame.emit(mark_img)
        self.update_log.emit(f"✅ 检测完成，耗时：{elapsed_time}ms")
        self.update_info.emit({
            "time": elapsed_time,
            "illegal": illegal_count,
            "legal": legal_count,
            "total": bike_count
        })
    
    def _process_batch_images(self):
        """处理批量图片"""
        for i, image_path in enumerate(self.source_path):
            if not self.running:
                break
            
            start_time = time.time()
            
            # 执行检测
            detect_res, det_img = bike_detect(image_path, conf=self.conf_thres)
            
            # 执行违规判断
            illegal_res, mark_img = judge_illegal(detect_res, det_img)
            
            # 保存违规记录
            save_status, save_res = save_illegal_record(illegal_res, mark_img, image_path, self.scene)
            
            # 计算耗时
            elapsed_time = int((time.time() - start_time) * 1000)
            
            # 统计信息
            bike_count = sum(1 for item in detect_res if item[0] == 0)  # 自行车数量
            illegal_count = len(illegal_res)  # 违规数量
            legal_count = bike_count - illegal_count  # 合法数量
            
            # 发送信号
            self.update_frame.emit(mark_img)
            self.update_log.emit(f"✅ 处理第 {i+1}/{len(self.source_path)} 张图片，耗时：{elapsed_time}ms")
            self.update_info.emit({
                "time": elapsed_time,
                "illegal": illegal_count,
                "legal": legal_count,
                "total": bike_count
            })
            
            # 短暂延时，让UI有时间更新
            time.sleep(0.1)
    
    def _process_video(self):
        """处理视频"""
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            self.update_log.emit("❌ 无法打开视频文件")
            return
        
        # 检查视频时长
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps if fps > 0 else 0
        
        if duration > 300:  # 超过5分钟
            self.show_video_duration_warning.emit()
            cap.release()
            return
        
        frame_count = 0
        start_time = time.time()
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 对每帧执行检测
            # 注意：这里为了速度，我们可以降低检测频率，比如每5帧检测一次
            if frame_count % 5 == 0:
                # 保存当前帧为临时图片
                temp_frame_path = "temp_frame.jpg"
                cv2.imwrite(temp_frame_path, frame)
                
                # 执行检测
                detect_res, det_img = bike_detect(temp_frame_path, conf=self.conf_thres)
                
                # 执行违规判断
                illegal_res, mark_img = judge_illegal(detect_res, det_img)
                
                # 保存违规记录
                save_status, save_res = save_illegal_record(illegal_res, mark_img, f"video_frame_{frame_count}", self.scene)
                
                # 发送信号
                self.update_frame.emit(mark_img)
                
            frame_count += 1
        
        cap.release()
        elapsed_time = int((time.time() - start_time) * 1000)
        self.update_log.emit(f"✅ 视频处理完成，耗时：{elapsed_time}ms")
    
    def _process_camera(self):
        """处理摄像头实时流"""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.update_log.emit("❌ 无法打开摄像头")
            return
        
        frame_count = 0
        start_time = time.time()
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 对每帧执行检测
            # 注意：这里为了速度，我们可以降低检测频率，比如每5帧检测一次
            if frame_count % 5 == 0:
                # 保存当前帧为临时图片
                temp_frame_path = "temp_frame.jpg"
                cv2.imwrite(temp_frame_path, frame)
                
                # 执行检测
                detect_res, det_img = bike_detect(temp_frame_path, conf=self.conf_thres)
                
                # 执行违规判断
                illegal_res, mark_img = judge_illegal(detect_res, det_img)
                
                # 保存违规记录
                save_status, save_res = save_illegal_record(illegal_res, mark_img, f"camera_frame_{frame_count}", self.scene)
                
                # 发送信号
                self.update_frame.emit(mark_img)
                
            frame_count += 1
        
        cap.release()
        elapsed_time = int((time.time() - start_time) * 1000)
        self.update_log.emit(f"✅ 摄像头检测完成，耗时：{elapsed_time}ms")
