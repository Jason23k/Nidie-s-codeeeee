# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 添加项目根目录

# from datasets.sliced_coco import SlicedCocoDetection
# from datasets.coco import make_coco_transforms

# # 假设你的数据路径如下（请按实际修改）
# img_folder = "E:/ultralytics/NIR/Videos_frames"
# ann_file = "E:/ultralytics/Nidie-s-codeeeee/VODkkkk/VOD-master/annotations/train.json"

# # dataset = SlicedCocoDetection(
# #     img_folder=img_folder,
# #     ann_file=ann_file,
# #     transforms=make_coco_transforms("train"),
# #     slice_size=800,
# #     small_obj_threshold=900
# # )

# dataset = SlicedCocoDetection(
#     img_folder=img_folder,
#     ann_file=ann_file,
#     transforms=make_coco_transforms("train"),
#     slice_size=640,
#     small_obj_threshold=100,
#     return_original_prob=0.0,  # 强制切片
# )

# img, target = dataset[0]
# # print("图像尺寸:", img.size)
# # print("目标数量:", len(target["boxes"]))
# # print("Boxes 示例:", target["boxes"][:2])

# print("✅ 成功加载切片样本！")
# print("图像类型:", type(img))
# print("boxes 形状:", target["boxes"].shape)
# print("labels:", target["labels"])

# test_sliced_datasets.py
# from datasets.sliced_coco import create_sliced_coco_dataset
# from torch.utils.data import DataLoader
# from datasets.coco import CocoDetection  # 假设你有这个类，类似 torchvision.datasets.CocoDetection

# def test_sliced_dataset():
#     # 配置路径（请替换成你的实际路径）
#     original_coco_json = "E:/ultralytics/NIR/Videos_frames"
#     original_image_dir = "E:/ultralytics/Nidie-s-codeeeee/VODkkkk/VOD-master/annotations/train.json"
#     sliced_output_dir = "./sliced_dataset"

#     # 1. 生成切片数据集
#     sliced_json, sliced_img_dir = create_sliced_coco_dataset(
#         coco_annotation_path=original_coco_json,
#         image_dir=original_image_dir,
#         output_dir=sliced_output_dir,
#         slice_height=640,
#         slice_width=640,
#         overlap_height_ratio=0.2,
#         overlap_width_ratio=0.2,
#         min_area_ratio=0.1,
#     )

#     # 2. 用你现有的 CocoDetection 加载切片后的数据
#     dataset = CocoDetection(
#         img_folder=sliced_img_dir,
#         ann_file=sliced_json,
#         transforms=None,  # 暂时不加 transform
#     )

#     # 3. 简单测试加载
#     print(f"切片后数据集大小: {len(dataset)}")
#     for i in range(min(3, len(dataset))):
#         img, target = dataset[i]
#         print(f"样本 {i}: 图像尺寸 {img.size}, 目标数 {len(target['boxes'])}")

# if __name__ == "__main__":
#     test_sliced_dataset()

# test_sliced_dataset.py
# test_sliced_dataset.py
from datasets.sliced_coco import create_sliced_coco_dataset
from datasets.coco import make_coco_transforms

def make_simple_transforms():
    return Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 尝试导入你的 CocoDetection
try:
    from datasets.coco import CocoDetection
except Exception:
    from torchvision.datasets import CocoDetection
    print("⚠️ 使用 torchvision 的 CocoDetection")

def test_sliced_dataset():
    # 🔴 替换为你的实际路径
    original_coco_json = "E:/ultralytics/Nidie-s-codeeeee/VODkkkk/VOD-master/annotations/train_clean.json"
    original_image_dir = "E:/ultralytics/NIR/Videos_frames"
    sliced_output_dir = "./sliced_dataset"

    # 生成切片数据集
    sliced_json, sliced_img_dir = create_sliced_coco_dataset(
        coco_annotation_path=original_coco_json,
        image_dir=original_image_dir,
        output_dir=sliced_output_dir,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        min_area_ratio=0.1,
    )

    # 加载（注意：图像在 sliced_img_dir 根目录，不是子文件夹）
    dataset = CocoDetection(
        img_folder=sliced_img_dir, 
        ann_file=sliced_json, 
        transforms=make_simple_transforms(),    # 调式成功后改为 make_coco_transforms("train") 或设为 None 测试原始 PIL 图像
        return_masks=False
    )

    print(f"📌 切片后数据集大小: {len(dataset)}")
    for i in range(min(3, len(dataset))):
        img, targets = dataset[i]
        num_boxes = len(targets) if isinstance(targets, list) else (
            len(targets["boxes"]) if "boxes" in targets else "N/A"
        )
        print(f"  样本 {i}: 图像尺寸 {img.size}, 目标数: {num_boxes}")

if __name__ == "__main__":
    test_sliced_dataset()