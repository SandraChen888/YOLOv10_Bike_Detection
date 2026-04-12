from ultralytics.utils.plotting import plot_results
import os
from pathlib import Path

# 项目根目录
root = Path(r"F:\YOLOv10_Bike_Detection\runs\detect")

# 遍历所有trainX文件夹
for train_dir in root.glob("train*"):
    if not train_dir.is_dir():
        continue
    csv_path = train_dir / "results.csv"
    if csv_path.exists():
        print(f"正在为 {train_dir.name} 生成训练图...")
        # 旧版本plot_results直接传入csv路径，自动保存在同目录下
        plot_results(str(csv_path))
        print(f"✅ {train_dir.name} 的results.png已生成！")

print("\n🎉 所有训练图已全部生成完成！")