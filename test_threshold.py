#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值敏感度测试脚本
用于确定违停判断的最佳重叠阈值
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detect_model import bike_detect
from illegal_judge import judge_illegal, calculate_overlap


class ThresholdTester:
    def __init__(self):
        """初始化测试器"""
        self.test_data_dir = "F:\\YOLOv10_Bike_Detection\\merged_bicycle_split\\test"
        self.results = {}
        
    def parse_yolo_label(self, label_path, img_width, img_height):
        """
        解析YOLO格式的标签文件
        :return: 自行车列表和黄格子列表
        """
        bikes = []
        yellow_grids = []
        
        if not os.path.exists(label_path):
            return bikes, yellow_grids
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                cls_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # 转换为[x1, y1, x2, y2]格式
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                if cls_id == 0:  # 自行车
                    bikes.append([x1, y1, x2, y2])
                elif cls_id == 1:  # 黄格子
                    yellow_grids.append([x1, y1, x2, y2])
        
        return bikes, yellow_grids
    
    def calculate_illegal_from_labels(self, bikes, yellow_grids, overlap_thresh):
        """
        根据标签计算真实的违规情况
        :return: 违规自行车列表
        """
        illegal_bikes = []
        
        for bike_box in bikes:
            is_illegal = True
            max_overlap = 0.0
            
            for grid_box in yellow_grids:
                overlap = calculate_overlap(bike_box, grid_box)
                if overlap > max_overlap:
                    max_overlap = overlap
            
            if max_overlap >= overlap_thresh:
                is_illegal = False
            
            if is_illegal:
                illegal_bikes.append(bike_box)
        
        return illegal_bikes
    
    def calculate_metrics(self, pred_illegal_count, true_illegal_count, total_bikes):
        """
        计算各项性能指标
        :param pred_illegal_count: 预测违规数
        :param true_illegal_count: 实际违规数
        :param total_bikes: 自行车总数
        """
        # 避免除零
        if true_illegal_count == 0:
            precision = 1.0 if pred_illegal_count == 0 else 0.0
            recall = 1.0 if pred_illegal_count == 0 else 0.0
        else:
            # 准确率：预测违规中正确的比例
            precision = min(pred_illegal_count, true_illegal_count) / pred_illegal_count if pred_illegal_count > 0 else 0.0
            # 召回率：实际违规中被检出的比例
            recall = min(pred_illegal_count, true_illegal_count) / true_illegal_count if true_illegal_count > 0 else 0.0
        
        # F1-Score
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        # 误报率：合法被误判为违规的比例
        false_positive_rate = (pred_illegal_count - min(pred_illegal_count, true_illegal_count)) / (total_bikes - true_illegal_count) if total_bikes - true_illegal_count > 0 else 0.0
        
        # 漏报率：违规被漏检的比例
        miss_rate = (true_illegal_count - min(pred_illegal_count, true_illegal_count)) / true_illegal_count if true_illegal_count > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "false_positive_rate": false_positive_rate,
            "miss_rate": miss_rate,
            "pred_illegal_count": pred_illegal_count,
            "true_illegal_count": true_illegal_count,
            "total_bikes": total_bikes
        }
    
    def test_threshold(self, overlap_thresh):
        """
        测试单个阈值
        """
        images_dir = os.path.join(self.test_data_dir, "images")
        labels_dir = os.path.join(self.test_data_dir, "labels")
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        total_pred_illegal = 0
        total_true_illegal = 0
        total_bikes = 0
        total_detect_time = 0
        image_count = 0
        
        for image_file in image_files[:50]:  # 限制测试数量，加快速度
            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            
            # 读取图像获取尺寸
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                continue
            img_height, img_width = img.shape[:2]
            
            # 解析真实标签
            true_bikes, true_yellow_grids = self.parse_yolo_label(label_path, img_width, img_height)
            
            # 目标检测
            start_time = time.time()
            detect_res, det_img = bike_detect(image_path, conf=0.5)
            detect_time = time.time() - start_time
            total_detect_time += detect_time
            
            # 分离检测结果
            pred_bikes = []
            pred_yellow_grids = []
            for cls_id, conf, x1, y1, x2, y2 in detect_res:
                if cls_id == 0:
                    pred_bikes.append([x1, y1, x2, y2])
                elif cls_id == 1:
                    pred_yellow_grids.append([x1, y1, x2, y2])
            
            # 计算真实违规
            true_illegal = self.calculate_illegal_from_labels(true_bikes, true_yellow_grids, overlap_thresh)
            
            # 计算预测违规
            pred_illegal, _ = judge_illegal(detect_res, img, overlap_thresh)
            pred_illegal_count = len(pred_illegal)
            
            total_pred_illegal += pred_illegal_count
            total_true_illegal += len(true_illegal)
            total_bikes += len(true_bikes)
            image_count += 1
        
        # 计算平均指标
        avg_detect_time = total_detect_time / image_count if image_count > 0 else 0
        
        metrics = self.calculate_metrics(total_pred_illegal, total_true_illegal, total_bikes)
        metrics["avg_detect_time"] = avg_detect_time
        metrics["test_images"] = image_count
        
        return metrics
    
    def run_threshold_sweep(self):
        """
        执行阈值扫描测试
        """
        thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        
        print("=" * 80)
        print("阈值敏感度测试")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试数据: {self.test_data_dir}")
        print("=" * 80)
        
        all_results = {}
        best_threshold = 0.65
        best_f1 = 0.0
        
        for thresh in thresholds:
            print(f"\n测试阈值: {thresh}...")
            
            try:
                metrics = self.test_threshold(thresh)
                all_results[thresh] = metrics
                
                print(f"  准确率: {metrics['precision']:.4f}")
                print(f"  召回率: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                print(f"  误报率: {metrics['false_positive_rate']:.4f}")
                print(f"  漏报率: {metrics['miss_rate']:.4f}")
                print(f"  平均检测时间: {metrics['avg_detect_time']:.4f}秒")
                
                # 更新最佳阈值
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    best_threshold = thresh
                    
            except Exception as e:
                print(f"  测试失败: {str(e)}")
                all_results[thresh] = {"error": str(e)}
        
        # 输出结果汇总
        print("\n" + "=" * 80)
        print("测试结果汇总")
        print("=" * 80)
        print(f"\n{'阈值':<8}{'准确率':<12}{'召回率':<12}{'F1-Score':<12}{'误报率':<12}{'漏报率':<12}{'检测时间':<12}")
        print("-" * 80)
        
        for thresh, metrics in all_results.items():
            if "error" not in metrics:
                print(f"{thresh:<8.2f}{metrics['precision']:<12.4f}{metrics['recall']:<12.4f}"
                      f"{metrics['f1_score']:<12.4f}{metrics['false_positive_rate']:<12.4f}"
                      f"{metrics['miss_rate']:<12.4f}{metrics['avg_detect_time']:<12.4f}")
        
        print("\n" + "=" * 80)
        print(f"推荐最佳阈值: {best_threshold} (F1-Score: {best_f1:.4f})")
        print("=" * 80)
        
        # 保存结果
        output_path = "threshold_test_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {output_path}")
        
        return all_results, best_threshold


if __name__ == "__main__":
    tester = ThresholdTester()
    tester.run_threshold_sweep()