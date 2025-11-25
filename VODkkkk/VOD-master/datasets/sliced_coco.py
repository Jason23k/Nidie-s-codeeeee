# # datasets/sliced_coco.py
# import torch
# import numpy as np
# from PIL import Image
# import os
# import random
# from sahi.slicing import slice_image
# from sahi.utils.coco import CocoAnnotation


# class SlicedCocoDetection(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         img_folder,
#         ann_file,
#         transforms,
#         slice_size=640,
#         overlap_ratio=0.2,
#         small_obj_threshold=900,
#         return_original_prob=0.3,
#         return_masks=False,
#     ):
#         from .coco import CocoDetection
#         # 初始化原始 coco 数据集（不传 transforms！）
#         self.coco_ds = CocoDetection(
#             img_folder=img_folder,
#             ann_file=ann_file,
#             transforms=None,          # ← 关键：不自动 transform
#             return_masks=return_masks
#         )
#         self.transforms = transforms
#         self.slice_size = slice_size
#         self.overlap_ratio = overlap_ratio
#         self.small_obj_threshold = small_obj_threshold
#         self.return_original_prob = return_original_prob

#     def __len__(self):
#         return len(self.coco_ds)

#     def __getitem__(self, idx):
#         # 获取原始数据（PIL Image + 原始 target）
#         original_img, original_target = self.coco_ds[idx]  # ← 这里 img 是 PIL！

#         # 决定是否使用原始图像 or 切片
#         if random.random() < self.return_original_prob:
#             img, target = original_img, original_target
#         else:
#             img, target = self._sahi_slice_sample(original_img, original_target)

#         # 最后统一应用 transforms
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)

#         return img, target

#     def _sahi_slice_sample(self, pil_img, target):
#         boxes = target["boxes"]
#         labels = target["labels"]
#         image_id = target["image_id"]

#         # 转 CocoAnnotation（xyxy）
#         coco_annotations = []
#         for box, label in zip(boxes, labels):
#             x1, y1, x2, y2 = box.tolist()
#             if x2 <= x1 or y2 <= y1:
#                 continue
#             ann = CocoAnnotation(
#                 bbox=[x1, y1, x2, y2],
#                 category_id=int(label),
#                 category_name="object"
#             )
#             coco_annotations.append(ann)

#         # 切片：关键！设置 image_dir=None 防止写磁盘（可选，但推荐）
#         slice_result = slice_image(
#             image=pil_img,
#             coco_annotation_list=coco_annotations,
#             slice_height=self.slice_size,
#             slice_width=self.slice_size,
#             overlap_height_ratio=self.overlap_ratio,
#             overlap_width_ratio=self.overlap_ratio
#             # keep_empty_slices=False,
#             # image_dir=None  # 👈 不保存图像到磁盘
#         )

#         # 现在：slice_result.images 和 slice_result.coco_images 长度相同，一一对应
#         if len(slice_result.coco_images) == 0:
#             return pil_img, target

#         # 找出非空切片的索引
#         valid_indices = [
#             i for i, ci in enumerate(slice_result.coco_images)
#             if len(ci.annotations) > 0
#         ]

#         if not valid_indices:
#             return pil_img, target

#         # 随机选一个有效索引
#         idx = random.choice(valid_indices)
#         sliced_img = slice_result.images[idx]          # PIL.Image ✅
#         sliced_anns = slice_result.coco_images[idx].annotations  # List[CocoAnnotation] ✅

#         # 过滤小目标
#         new_boxes = []
#         new_labels = []
#         for ann in sliced_anns:
#             bbox = ann.bbox
#             w = bbox[2] - bbox[0]
#             h = bbox[3] - bbox[1]
#             if w * h >= self.small_obj_threshold:
#                 new_boxes.append(bbox)
#                 new_labels.append(ann.category_id)

#         if not new_boxes:
#             return pil_img, target

#         new_target = {
#             "image_id": image_id,
#             "boxes": torch.as_tensor(new_boxes, dtype=torch.float32),
#             "labels": torch.as_tensor(new_labels, dtype=torch.int64),
#             "orig_size": torch.as_tensor([sliced_img.height, sliced_img.width]),
#             "size": torch.as_tensor([sliced_img.height, sliced_img.width]),
#         }

#         return sliced_img, new_target

# create_sliced_coco.py
# import os
# from sahi.utils.coco import Coco
# from sahi.utils.file import save_json
# from sahi.slicing import slice_coco

# def create_sliced_coco_dataset(
#     coco_annotation_path: str,
#     image_dir: str,
#     output_dir: str,
#     slice_height: int = 640,
#     slice_width: int = 640,
#     overlap_height_ratio: float = 0.2,
#     overlap_width_ratio: float = 0.2,
#     min_area_ratio: float = 0.1,
#     verbose: bool = True
# ):
#     """
#     使用 SAHI 对 COCO 数据集进行切片，生成新的 COCO 格式数据集。
    
#     Args:
#         coco_annotation_path: 原始 COCO JSON 路径
#         image_dir: 原始图像目录
#         output_dir: 切片输出目录（会创建 images/ 和 annotations.json）
#         slice_height/width: 切片尺寸
#         overlap_*_ratio: 重叠比例 [0, 1)
#         min_area_ratio: 小于该比例的标注会被过滤（防止切碎小目标）
#     """
#     # 创建输出目录
#     sliced_image_dir = os.path.join(output_dir, "images")
#     os.makedirs(sliced_image_dir, exist_ok=True)

#     # 执行切片
#     coco_dict, _ = slice_coco(
#         coco_annotation_file_path=coco_annotation_path,
#         image_dir=image_dir,
#         output_dir=sliced_image_dir,
#         slice_height=slice_height,
#         slice_width=slice_width,
#         overlap_height_ratio=overlap_height_ratio,
#         overlap_width_ratio=overlap_width_ratio,
#         min_area_ratio=min_area_ratio,
#         verbose=verbose,
#     )

#     # 保存新的 COCO JSON
#     output_json_path = os.path.join(output_dir, "annotations.json")
#     save_json(coco_dict, output_json_path)

#     print(f"✅ 切片完成！\n - 图像: {sliced_image_dir}\n - 标注: {output_json_path}")
#     return output_json_path, sliced_image_dir

# create_sliced_coco.py
# create_sliced_coco.py (for sahi==0.11.36)
import os
from sahi.slicing import slice_coco

def create_sliced_coco_dataset(
    coco_annotation_path: str,
    image_dir: str,
    output_dir: str,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
    verbose: bool = True
):
    """
    专为 SAHI v0.11.36 设计的 COCO 切片函数。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 调用 slice_coco（注意参数顺序和含义）
    coco_dict, _ = slice_coco(
        coco_annotation_file_path=coco_annotation_path,
        image_dir=image_dir,
        output_coco_annotation_file_name="annotations.json",  # 只是文件名！
        output_dir=output_dir,  # 图像和 JSON 都会放在这里
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        min_area_ratio=min_area_ratio,
        verbose=verbose,
    )

    # SAHI 0.11.36 会自动把 annotations.json 写入 output_dir
    sliced_image_dir = output_dir  # 因为图像也直接存到 output_dir 根下
    output_json_path = os.path.join(output_dir, "annotations.json")

    print(f"✅ SAHI 0.11.36 切片完成！\n - 图像 & 标注目录: {output_dir}")
    return output_json_path, sliced_image_dir