from ultralytics.utils.plotting import plot_results
import os

# 👇 这里改成你自己的results.csv路径，就是你train6文件夹里的那个
csv_path = r"F:\YOLOv10_Bike_Detection\runs\detect\train6\results.csv"

# 调用YOLO原生绘图函数，生成和官方完全一致的10子图
plot_results(csv_path)

# 打印生成路径，方便你找图
save_path = os.path.join(os.path.dirname(csv_path), "results.png")
print(f"✅ 图表已成功生成！路径：{save_path}")
print("📊 图表样式和YOLO原生训练图完全一致，包含10个子图：损失+指标曲线")