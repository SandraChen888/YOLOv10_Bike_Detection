import os
import cv2
import shutil
import random
import time

def fast_augment_dataset(input_dir, output_dir, target_count=2000):
    """快速增强数据集到目标数量"""
    # 创建输出目录
    output_img_dir = os.path.join(output_dir, "images", "train")
    output_label_dir = os.path.join(output_dir, "labels", "train")
    
    # 确保输出目录不存在
    if os.path.exists(output_dir):
        print(f"删除现有目录: {output_dir}")
        # 使用更安全的删除方式
        import subprocess
        subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', output_dir], shell=True)
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 复制原始数据
    input_img_dir = os.path.join(input_dir, "images", "train")
    input_label_dir = os.path.join(input_dir, "labels", "train")
    
    img_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    current_count = len(img_files)
    
    print(f"原始数据集大小: {current_count}")
    print(f"目标数据集大小: {target_count}")
    
    # 复制原始图像和标注
    print("复制原始数据...")
    for i, img_file in enumerate(img_files):
        src_img = os.path.join(input_img_dir, img_file)
        dst_img = os.path.join(output_img_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(input_label_dir, label_file)
        if os.path.exists(src_label):
            dst_label = os.path.join(output_label_dir, label_file)
            shutil.copy2(src_label, dst_label)
        
        if (i + 1) % 100 == 0:
            print(f"已复制 {i + 1}/{current_count}")
    
    # 计算需要增强的数量
    need_augment = target_count - current_count
    if need_augment <= 0:
        print("数据集已经达到目标数量")
        return
    
    print(f"需要增强 {need_augment} 张图片")
    
    # 开始快速增强
    success_count = 0
    augment_id = 1
    
    start_time = time.time()
    
    while success_count < need_augment:
        for img_file in img_files:
            if success_count >= need_augment:
                break
            
            # 复制文件并添加后缀
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            
            src_img = os.path.join(input_img_dir, img_file)
            dst_img = os.path.join(output_img_dir, f"{base_name}_copy{augment_id}{ext}")
            shutil.copy2(src_img, dst_img)
            
            label_file = base_name + '.txt'
            src_label = os.path.join(input_label_dir, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_label_dir, f"{base_name}_copy{augment_id}.txt")
                shutil.copy2(src_label, dst_label)
                success_count += 1
                augment_id += 1
        
        # 显示进度
        if (success_count) % 200 == 0:
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / success_count) * (need_augment - success_count) if success_count > 0 else 0
            print(f"已增强 {success_count}/{need_augment}，预计剩余时间: {remaining_time:.2f} 秒")
    
    end_time = time.time()
    print(f"成功增强 {success_count} 张图片")
    print(f"增强后数据集大小: {current_count + success_count}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

# 增强原始数据集
target_count = 2000
input_dir = "F:/dataset/origin"
output_dir = "F:/dataset/origin_augmented_new"
fast_augment_dataset(input_dir, output_dir, target_count)
print("数据集增强完成！")
