from ultralytics import YOLO
import os

# ===================== 本地路径配置（完全匹配你的目录）=====================
BASE_DIR = r"F:\YOLOv10_Bike_Detection\merged_bicycle_split"
MODEL_PATH = os.path.join(BASE_DIR, r"runs\detect\bike_yellow_grid_300e\weights\best.pt")
YAML_PATH = os.path.join(BASE_DIR, "bicycle_grid.yaml")
TEST_IMG_PATH = os.path.join(BASE_DIR, r"test\images")

if __name__ == '__main__':
    # 加载模型
    model = YOLO(MODEL_PATH)

    print("=" * 60)
    print("开始验证模型精度...")
    print("=" * 60)

    # 验证：强制使用本地绝对路径，彻底解决路径不匹配
    metrics = model.val(
        data=YAML_PATH,
        imgsz=640,
        batch=4,
        device=0,
        plots=True,
        project=os.path.join(BASE_DIR, "runs"),  # 结果保存在本地项目目录
        name="val_result"
    )

    print("\n✅ 验证完成！")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"bike 精度: {metrics.box.p[0]:.4f}")
    print(f"yellow_grid 精度: {metrics.box.p[1]:.4f}")

    print("\n开始对 test 集推理...")

    # 推理
    model.predict(
        source=TEST_IMG_PATH,
        imgsz=640,
        device=0,
        save=True,
        conf=0.25,
        project=os.path.join(BASE_DIR, "runs"),
        name="test_result"
    )

    print("\n🎉 全部完成！")
    print(f"📁 验证图表保存在: {os.path.join(BASE_DIR, 'runs', 'val_result')}")
    print(f"📁 推理结果保存在: {os.path.join(BASE_DIR, 'runs', 'test_result')}")