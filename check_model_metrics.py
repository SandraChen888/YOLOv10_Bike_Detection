import pandas as pd

# 读取训练结果文件（使用性能最好的train6目录）
df = pd.read_csv("F:/YOLOv10_Bike_Detection/runs/detect/train_new/results.csv")
df.columns = df.columns.str.strip()  # 去除所有空格

# 提取最终epoch的所有指标
final = df.iloc[-1]

print("="*60)
print("你的 YOLOv10 单车检测模型 最终真实指标")
print("="*60)
print(f"mAP@0.5 (平均精度均值): {final['metrics/mAP50(B)']:.4f}")
print(f"mAP@0.5:0.95 (综合精度): {final['metrics/mAP50-95(B)']:.4f}")
print(f"Precision (精确率): {final['metrics/precision(B)']:.4f}")
print(f"Recall (召回率): {final['metrics/recall(B)']:.4f}")
print("-" * 60)
print(f"训练损失 (train/box_loss): {final['train/box_loss']:.4f}")
print(f"训练分类损失 (train/cls_loss): {final['train/cls_loss']:.4f}")
print(f"训练DFL损失 (train/dfl_loss): {final['train/dfl_loss']:.4f}")
print("-" * 60)
print(f"验证损失 (val/box_loss): {final['val/box_loss']:.4f}")
print(f"验证分类损失 (val/cls_loss): {final['val/cls_loss']:.4f}")
print(f"验证DFL损失 (val/dfl_loss): {final['val/dfl_loss']:.4f}")
print("="*60)

# 额外：打印训练过程的最佳mAP@0.5
best_idx = df['metrics/mAP50(B)'].idxmax()
best_map = df.loc[best_idx, 'metrics/mAP50(B)']
print(f"\n🏆 训练过程最佳 mAP@0.5: {best_map:.4f}")