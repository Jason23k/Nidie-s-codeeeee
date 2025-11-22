import json
import os

def validate_coco_json(json_path):
    print(f"🔍 正在验证: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 检查必要字段
    required_keys = ['images', 'annotations', 'categories', 'videos']
    for key in required_keys:
        if key not in data:
            print(f"❌ 缺少字段: {key}")
            return False

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    videos = data['videos']

    print(f"✅ 包含 {len(images)} 张图片, {len(annotations)} 个标注, {len(categories)} 个类别, {len(videos)} 个视频")

    # 检查 image_id 是否在 images 中
    image_ids = set(img['id'] for img in images)
    for ann in annotations:
        if ann['image_id'] not in image_ids:
            print(f"❌ 标注 {ann['id']} 的 image_id {ann['image_id']} 不存在于 images 中")
            return False

    # 检查 video_id 是否在 videos 中
    video_ids = set(video['id'] for video in videos)
    for img in images:
        if img['video_id'] not in video_ids:
            print(f"❌ 图片 {img['id']} 的 video_id {img['video_id']} 不存在于 videos 中")
            return False

    print("✅ JSON 文件格式正确！")
    return True

# ====== 主程序 ======
if __name__ == "__main__":
    base_dir = r"D:\广财\F's research\ShipsVideos\NIR"
    train_json = os.path.join(base_dir, "ships_nir_train.json")
    val_json = os.path.join(base_dir, "ships_nir_val.json")

    print("🟢 验证训练集...")
    validate_coco_json(train_json)

    print("\n🟢 验证验证集...")
    validate_coco_json(val_json)