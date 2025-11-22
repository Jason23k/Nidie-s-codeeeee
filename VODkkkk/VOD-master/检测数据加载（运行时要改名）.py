from datasets.coco import build
import argparse

class Args:
    dataset_file = "ships"
    masks = False
    cache_mode = False

args = Args()
dataset = build("train", args)
print("✅ Dataset 构建成功！")
print("📌 示例图片路径:", dataset[0][0].shape)  # img 是 Tensor
print("📌 示例 target keys:", list(dataset[0][1].keys()))