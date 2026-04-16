#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统测试脚本
用于验证毕业设计的功能完整性和性能指标
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
import cv2

# 导入系统模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detect_model import bike_detect
from illegal_judge import judge_illegal
from record_manage import query_record, save_detection_record

class SystemTester:
    def __init__(self):
        """初始化测试器"""
        self.test_results = {
            "功能测试": {},
            "性能测试": {}
        }
        self.test_data_dir = "F:\\YOLOv10_Bike_Detection\\merged_bicycle_split\\test"
        self.model_path = "F:\\YOLOv10_Bike_Detection\\models\\best.pt"
    
    def test_functionality(self):
        """功能测试"""
        print("\n=== 功能测试 ===")
        
        # 1. 测试数据采集与预处理
        print("\n1. 测试数据采集与预处理...")
        try:
            # 检查测试数据是否存在
            images_dir = os.path.join(self.test_data_dir, "images")
            labels_dir = os.path.join(self.test_data_dir, "labels")
            
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                
                if len(image_files) > 0 and len(label_files) > 0:
                    self.test_results["功能测试"]["数据采集与预处理"] = "通过"
                    print(f"   ✓ 数据目录存在，包含 {len(image_files)} 张图像和 {len(label_files)} 个标签文件")
                else:
                    self.test_results["功能测试"]["数据采集与预处理"] = "失败"
                    print("   ✗ 数据目录存在但缺少文件")
            else:
                self.test_results["功能测试"]["数据采集与预处理"] = "失败"
                print("   ✗ 数据目录不存在")
        except Exception as e:
            self.test_results["功能测试"]["数据采集与预处理"] = f"失败: {str(e)}"
            print(f"   ✗ 发生错误: {str(e)}")
        
        # 2. 测试目标检测功能
        print("\n2. 测试目标检测功能...")
        try:
            # 选择一张测试图像
            test_image = os.path.join(self.test_data_dir, "images", os.listdir(os.path.join(self.test_data_dir, "images"))[0])
            
            # 直接使用bike_detect函数进行检测
            detect_res, det_img = bike_detect(test_image, conf=0.5)
            
            if detect_res:
                self.test_results["功能测试"]["目标检测"] = "通过"
                print(f"   ✓ 目标检测功能正常，检测到 {len(detect_res)} 个目标")
            else:
                self.test_results["功能测试"]["目标检测"] = "通过 (未检测到目标)"
                print("   ✓ 目标检测功能正常，未检测到目标")
        except Exception as e:
            self.test_results["功能测试"]["目标检测"] = f"失败: {str(e)}"
            print(f"   ✗ 发生错误: {str(e)}")
        
        # 3. 测试违规判断功能
        print("\n3. 测试违规判断功能...")
        try:
            # 选择一张测试图像
            test_image = os.path.join(self.test_data_dir, "images", os.listdir(os.path.join(self.test_data_dir, "images"))[0])
            
            # 先进行检测
            detect_res, det_img = bike_detect(test_image, conf=0.5)
            
            # 测试违规判断
            illegal_res, mark_img = judge_illegal(detect_res, det_img, 0.65)
            
            self.test_results["功能测试"]["违规判断"] = "通过"
            print(f"   ✓ 违规判断功能正常，检测到 {len(illegal_res)} 个违规目标")
        except Exception as e:
            self.test_results["功能测试"]["违规判断"] = f"失败: {str(e)}"
            print(f"   ✗ 发生错误: {str(e)}")
        
        # 4. 测试记录管理功能
        print("\n4. 测试记录管理功能...")
        try:
            # 测试数据库连接和查询
            records = query_record()
            self.test_results["功能测试"]["记录管理"] = "通过"
            print(f"   ✓ 记录管理功能正常，当前有 {len(records)} 条记录")
        except Exception as e:
            self.test_results["功能测试"]["记录管理"] = f"失败: {str(e)}"
            print(f"   ✗ 发生错误: {str(e)}")
    
    def test_performance(self):
        """性能测试"""
        print("\n=== 性能测试 ===")
        
        # 1. 测试检测速度
        print("\n1. 测试检测速度...")
        try:
            # 选择10张测试图像
            image_files = [f for f in os.listdir(os.path.join(self.test_data_dir, "images")) if f.endswith(('.jpg', '.jpeg', '.png'))][:10]
            total_time = 0
            
            for image_file in image_files:
                test_image = os.path.join(self.test_data_dir, "images", image_file)
                
                # 记录开始时间
                start_time = time.time()
                
                # 执行检测
                detect_res, det_img = bike_detect(test_image, conf=0.5)
                
                # 记录结束时间
                end_time = time.time()
                total_time += (end_time - start_time)
            
            avg_time = total_time / len(image_files)
            self.test_results["性能测试"]["检测速度"] = f"{avg_time:.4f}秒/张"
            print(f"   ✓ 平均检测速度: {avg_time:.4f}秒/张")
        except Exception as e:
            self.test_results["性能测试"]["检测速度"] = f"失败: {str(e)}"
            print(f"   ✗ 发生错误: {str(e)}")
        
        # 2. 测试记录查询速度
        print("\n2. 测试记录查询速度...")
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 测试获取记录
            records = query_record()
            
            # 记录结束时间
            end_time = time.time()
            query_time = end_time - start_time
            
            self.test_results["性能测试"]["记录查询速度"] = f"{query_time:.4f}秒"
            print(f"   ✓ 记录查询速度: {query_time:.4f}秒")
        except Exception as e:
            self.test_results["性能测试"]["记录查询速度"] = f"失败: {str(e)}"
            print(f"   ✗ 发生错误: {str(e)}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始系统测试...")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 运行功能测试
        self.test_functionality()
        
        # 运行性能测试
        self.test_performance()
        
        # 输出测试结果
        print("\n=== 测试结果 ===")
        print(json.dumps(self.test_results, ensure_ascii=False, indent=2))
        
        # 保存测试结果
        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print("\n测试结果已保存到 test_results.json")

if __name__ == "__main__":
    tester = SystemTester()
    tester.run_all_tests()
