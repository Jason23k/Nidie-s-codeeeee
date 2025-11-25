# debug_sliced_loader.py
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径（根据你的结构调整）
sys.path.append(str(Path(__file__).parent))  # 假设脚本在项目根目录

from datasets.coco import CocoDetection, make_coco_transforms
from datasets.transforms_single import Compose, ToTensor, Normalize


def make_simple_transforms():
    return Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def debug_sliced_dataset():
    sliced_img_dir = "E:/ultralytics/Nidie-s-codeeeee/VODkkkk/VOD-master/sliced_dataset"
    sliced_json = "E:/ultralytics/Nidie-s-codeeeee/VODkkkk/VOD-master/sliced_dataset/annotations.json_coco.json"

    print(f"🔍 检查路径是否存在...")
    assert os.path.exists(sliced_img_dir), f"图像目录不存在: {sliced_img_dir}"
    assert os.path.exists(sliced_json), f"标注文件不存在: {sliced_json}"

    print("✅ 路径存在，开始加载数据集...")

    # 关键：使用正确的参数名 ann_file（全小写！）
    dataset = CocoDetection(
        img_folder=sliced_img_dir,
        ann_file=sliced_json,
        transforms=make_coco_transforms('train'),  # 或设为 None 测试原始 PIL 图像
        return_masks=False
    )

    print(f"🎉 数据集加载成功！总样本数: {len(dataset)}\n")

    # 随机查看前 3 个样本
    for i in range(min(10, len(dataset))):
        try:
            img, target = dataset[i]
            print(f"=== 样本 {i} ===")
            print(f"图像类型: {type(img)}")
            print(f"图像形状: {img.shape if hasattr(img, 'shape') else img.size}")
            print(f"image_id: {target['image_id'].item()}")
            print(f"目标数量: {len(target['boxes'])}")
            if len(target['boxes']) > 0:
                print(f"第一个框 (x1,y1,x2,y2): {target['boxes'][0].tolist()}")
                print(f"对应类别 ID: {target['labels'][0].item()}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ 样本 {i} 加载失败: {e}")
            raise


if __name__ == "__main__":
    debug_sliced_dataset()