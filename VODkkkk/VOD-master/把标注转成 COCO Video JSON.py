import os
import json
import random
from scipy.io import loadmat

# ====== 配置区 ======
base_dir = r"D:\广财\F's research\ShipsVideos\NIR"
video_dir = os.path.join(base_dir, "Videos")
anno_dir = os.path.join(base_dir, "ObjectGT")

train_ratio = 0.8
random.seed(42)

# ====== 获取视频列表 ======
video_names = [
    f for f in os.listdir(video_dir)
    if f.endswith((".mp4", ".avi"))
]
print(f"✅ 找到 {len(video_names)} 个视频")

random.shuffle(video_names)
split_idx = int(len(video_names) * train_ratio)
train_videos = video_names[:split_idx]
val_videos = video_names[split_idx:]

print(f"📌 训练集: {len(train_videos)} 个视频")
print(f"📌 验证集: {len(val_videos)} 个视频")


def build_coco_json(video_list, output_json):
    categories = [{"id": 1, "name": "ship"}]
    images = []
    annotations = []
    videos = []

    video_id = 1
    image_id = 1
    ann_id = 1

    for video_name in video_list:
        # 构造 .mat 文件名
        mat_name = video_name.replace(".mp4", "_ObjectGT.mat").replace(".avi", "_ObjectGT.mat")
        video_anno_file = os.path.join(anno_dir, mat_name)

        if not os.path.exists(video_anno_file):
            print(f"⚠️ 无标注: {video_anno_file}")
            continue

        try:
            mat_data = loadmat(video_anno_file)
        except Exception as e:
            print(f"❌ 无法加载 {video_anno_file}: {e}")
            continue

        if 'structXML' not in mat_data:
            print(f"❌ 无 'structXML' 字段: {video_anno_file}")
            continue

        structXML = mat_data['structXML']
        if structXML.size == 0:
            print(f"❌ 'structXML' 为空: {video_anno_file}")
            continue

        bb_data = structXML['BB'][0]  # shape: (N,)
        width, height = 1920, 1080

        videos.append({
            "id": video_id,
            "file_name": video_name,
            "width": width,
            "height": height
        })

        for frame_idx in range(len(bb_data)):
            bb = bb_data[frame_idx]
            if bb.size == 0:
                continue

            frame_id = frame_idx + 1

            # 处理单船 or 多船
            if bb.ndim == 2:
                ship_boxes = bb
            elif bb.ndim == 1:
                ship_boxes = [bb]
            else:
                continue

            # ✅ 为本帧添加 image（每帧只加一次）
            img_name = f"{video_name}_frame{frame_id:06d}.jpg"
            images.append({
                "id": image_id,
                "video_id": video_id,
                "frame_id": frame_id,
                "file_name": img_name,
                "height": height,
                "width": width,
            })

            # ✅ 遍历该帧中的每一艘船
            for ship_box in ship_boxes:
                try:
                    x = float(ship_box[0])
                    y = float(ship_box[1])
                    w = float(ship_box[2])
                    h = float(ship_box[3])
                    class_id = 1

                    # ✅ 【关键修复】轻微越界 bbox 不跳过，而是 clip 到边界
                    x = max(0.0, x)
                    y = max(0.0, y)
                    w = max(1.0, w)  # 宽高至少为 1
                    h = max(1.0, h)

                    # 跳过明显无效 bbox（如 w > width）
                    if w > width * 1.1 or h > height * 1.1:
                        continue

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    ann_id += 1

                except Exception as e:
                    print(f"❌ 解析 bbox 失败: {e} | 视频: {video_name} | 帧: {frame_id} | BB: {ship_box}")

            image_id += 1

        video_id += 1

    # ✅ 【关键修复】定义 coco_data！
    coco_data = {
        "categories": categories,
        "videos": videos,
        "images": images,
        "annotations": annotations
    }

    # 保存到你的数据目录
    output_path = os.path.join(base_dir, output_json)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"✅ 已生成: {output_path} | 图片: {len(images)}, 标注: {len(annotations)}")


# ====== 主程序 ======
if __name__ == "__main__":
    build_coco_json(train_videos, "ships_nir_train.json")
    build_coco_json(val_videos, "ships_nir_val.json")
    print("🎉 全部完成！")