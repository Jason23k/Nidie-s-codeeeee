import os
import cv2
import numpy as np
import scipy.io as sio
from pathlib import Path
import yaml
import subprocess
import sys
from glob import glob

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # 可选，防 Qt 后端干扰
import matplotlib
matplotlib.use('Agg')  # 👈 关键：强制使用非 GUI 后端



# ==================== 【请按你的实际路径修改】====================
VIDEO_DIR = r"D:\广财\F's research\ShipsVideos\VIS_Onshore\Videos"
OBJGT_DIR = r"D:\广财\F's research\ShipsVideos\VIS_Onshore\ObjectGT"

# 输出目录（会自动创建）
OUTPUT_ROOT = r"D:\YOLO\VIS_Onshore_YOLO"

# 模型权重路径（替换为你训练好的 best.pt）
WEIGHTS_PATH = r"D:\YOLO\ultralytics-8.3.26\SeaShips_SMD\mini_train\weights\best.pt"

# 类别名（必须与你训练时的 data.yaml 严格一致！）
NAMES = [
    'vessel', 'Speed boat', 'Other', 'Sail boat', 'Ferry',
    'general cargo ship', 'container ship', 'Buoy', 'flying bird',
    'fishing boat', 'bulk cargo carrier', 'ore carrier', 'Boat',
    'passenger ship', 'Kayak'
]
NC = len(NAMES)

# 抽帧率：1 fps（每秒1帧）；可设为 0.5（每2秒1帧）以减少数据量
FPS_TARGET = 1.0

# 是否保留无目标帧（生成空 .txt）？建议 False（仅评估有目标帧）
KEEP_EMPTY_FRAMES = False


# ==================== 配置结束 ====================


def find_field(names, candidates):
    for cand in candidates:
        if cand in names:
            return cand
    raise ValueError(f"字段未找到: {candidates} ∉ {names}")


def load_mat_annotations(mat_path):
    try:
        mat_data = sio.loadmat(mat_path, simplify_cells=False)
        if 'structXML' not in mat_data:
            raise ValueError("❌ 未找到 'structXML' 字段")

        struct_xml = mat_data['structXML']  # (1, N)
        n_frames = struct_xml.shape[1]
        print(f"  📁 共 {n_frames} 帧")

        all_frames = []
        all_bboxes = []
        all_cls_ids = []

        for frame_idx in range(n_frames):
            frame_struct = struct_xml[0, frame_idx]
            bb_field = 'BB'
            obj_type_field = 'ObjectType'

            if bb_field not in frame_struct.dtype.names or obj_type_field not in frame_struct.dtype.names:
                continue

            BB = frame_struct[bb_field]  # shape: (k, ?)
            ObjectType = frame_struct[obj_type_field]  # shape: (k, 1)

            if BB.size == 0:
                continue

            # 🔑 核心修复：遍历每个目标，安全取 bbox
            for i in range(BB.shape[0]):
                # --- bbox 解析 ---
                bbox_raw = BB[i]
                if not isinstance(bbox_raw, np.ndarray) or bbox_raw.size == 0:
                    continue
                bbox_vals = bbox_raw.flatten()

                # ✅ 关键：根据长度自适应解析
                if len(bbox_vals) >= 4:
                    x, y, w, h = bbox_vals[:4]
                elif len(bbox_vals) == 3:
                    # 假设 [x1, y1, w]，用 w * 0.5 估算 h（船舶宽高比≈2）
                    x, y, w = bbox_vals
                    h = w * 0.5  # ⚠️ 估算！后续可校准
                elif len(bbox_vals) == 2:
                    # 只有中心点 → 跳过
                    continue
                else:
                    continue

                if w <= 5 or h <= 5:  # 过滤极小框
                    continue

                # --- 类别解析 ---
                cls_name = 'Other'
                try:
                    raw = ObjectType[i, 0]
                    if hasattr(raw, 'item'):
                        cls_name = raw.item()
                    else:
                        cls_name = str(raw).strip()
                except:
                    pass

                # 映射到 NAMES（严格匹配 + 模糊匹配）
                cls_id = 2  # default: 'Other'
                for idx, name in enumerate(NAMES):
                    if name.lower().replace(' ', '') in cls_name.lower().replace(' ', '').replace('/', ''):
                        cls_id = idx
                        break

                all_frames.append(frame_idx + 1)
                all_bboxes.append([x, y, w, h])  # ✅ 统一 xywh
                all_cls_ids.append(cls_id)

        if not all_frames:
            print("  ⚠️  无有效标注")
            return None, None, None

        print(f"  ✅ 解析成功: {len(all_frames)} 帧, {len(all_bboxes)} 个目标")
        return (
            np.array(all_frames, dtype=int),
            np.array(all_bboxes, dtype=float),
            np.array(all_cls_ids, dtype=int)
        )

    except Exception as e:
        import traceback
        print(f"  ❌ 解析失败: {e}")
        traceback.print_exc()
        return None, None, None


def convert_bbox_to_yolo(x, y, w, h, img_w, img_h):
    """输入 xywh → 输出 cx, cy, rw, rh（归一化）"""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    rw = w / img_w
    rh = h / img_h
    # 限幅防越界
    cx = np.clip(cx, 0, 1)
    cy = np.clip(cy, 0, 1)
    rw = np.clip(rw, 0, 1)
    rh = np.clip(rh, 0, 1)
    return cx, cy, rw, rh


def main():
    print("=" * 70)
    print("🌊 VIS_Onshore 视频目标检测 baseline 评估（严格单帧模式）")
    print("🎯 目标：复现文献原模型性能（无任何改进）")
    print("=" * 70)

    # 创建输出目录
    output_img_dir = Path(OUTPUT_ROOT) / "images" / "val"
    output_lbl_dir = Path(OUTPUT_ROOT) / "labels" / "val"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 输出目录: {OUTPUT_ROOT}")
    print(f"📽️  视频目录: {VIDEO_DIR}")

    total_images = 0
    total_labels = 0
    skipped_videos = []

    # 支持的视频格式
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}

    # 遍历所有视频
    video_files = [f for f in Path(VIDEO_DIR).iterdir() if f.suffix.lower() in VIDEO_EXTS]
    print(f"\n🔍 找到 {len(video_files)} 个视频，开始处理...")

    for video_path in sorted(video_files):
        video_name = video_path.stem
        mat_path = Path(OBJGT_DIR) / f"{video_name}_ObjectGT.mat"

        if not mat_path.exists():
            print(f"❌ 跳过 {video_name}: {mat_path.name} 不存在")
            skipped_videos.append(video_name)
            continue

        print(f"\n📦 处理 {video_name}")

        # 加载标注
        frames, bboxes, cls_ids = load_mat_annotations(str(mat_path))
        if frames is None:
            skipped_videos.append(video_name)
            continue

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠️  无法打开视频")
            cap.release()
            skipped_videos.append(video_name)
            continue

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"   原始: {total_video_frames} 帧, {orig_fps:.2f} fps → 目标抽帧率: {FPS_TARGET} fps")

        # 计算抽帧间隔（按时间，非帧号）
        frame_interval = max(1, int(round(orig_fps / FPS_TARGET)))

        frame_idx = 0
        saved_images = 0
        saved_labels = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检查是否该帧需要处理（按时间间隔）
            if frame_idx % frame_interval == 0:
                target_frame_num = frame_idx + 1  # 标注帧号从1开始

                # 查找该帧的所有标注
                mask = (frames == target_frame_num)
                frame_bboxes = bboxes[mask]
                frame_cls_ids = cls_ids[mask]

                # 保存图像
                img_filename = f"{video_name}_f{frame_idx:05d}.jpg"
                img_path = output_img_dir / img_filename
                cv2.imwrite(str(img_path), frame)
                saved_images += 1

                # 保存标签
                lbl_filename = img_filename.replace('.jpg', '.txt')
                lbl_path = output_lbl_dir / lbl_filename

                img_h, img_w = frame.shape[:2]
                with open(lbl_path, 'w') as f:
                    for bbox, cls_id in zip(frame_bboxes, frame_cls_ids):
                        x, y, w, h = bbox
                        cx, cy, rw, rh = convert_bbox_to_yolo(x, y, w, h, img_w, img_h)
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {rw:.6f} {rh:.6f}\n")
                        saved_labels += 1

                # 若无目标且不保留空帧，则删除空 .txt（但已保存 image）
                if saved_labels == 0 and not KEEP_EMPTY_FRAMES:
                    lbl_path.unlink(missing_ok=True)
                    # 注意：image 已保存，若想删 image 需额外处理

            frame_idx += 1

        cap.release()
        print(f"   ✅ 保存 {saved_images} 帧图像, {saved_labels} 个目标标注")
        total_images += saved_images
        total_labels += saved_labels

    print(f"\n🎉 总计: {total_images} 张图像, {total_labels} 个目标")
    if skipped_videos:
        print(f"⚠️  跳过视频 ({len(skipped_videos)}): {', '.join(skipped_videos)}")

    # 生成 data_vis.yaml
    data_yaml = {
        "path": OUTPUT_ROOT.replace("\\", "/"),
        "train": "",  # 仅评估，train 可为空
        "val": "images/val",
        "nc": NC,
        "names": NAMES
    }

    yaml_path = Path(OUTPUT_ROOT) / "data_vis.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)
    print(f"\n📝 已生成配置文件: {yaml_path}")

    # ==================== 运行评估 ====================
    print("\n🚀 启动 YOLOv8 单帧检测评估...")

    # 检查权重是否存在
    if not Path(WEIGHTS_PATH).exists():
        print(f"❌ 权重文件不存在: {WEIGHTS_PATH}")
        print("请修改脚本中的 WEIGHTS_PATH 为你训练好的 best.pt 路径")
        return False

    cmd_str = (
        f"yolo val "
        f"model=\"{WEIGHTS_PATH}\" "
        f"data=\"{yaml_path}\" "
        f"imgsz=640 "
        f"batch=8 "
        f"name=eval_vis_onshore_baseline "
        f"save_json=True "
        f"plots=True "
        f"save=True " 
        f"exist_ok=True"
    )

    print(f"   命令: {cmd_str}")
    print("   （正在运行评估，请稍候...）")

    # 直接执行，不捕获输出 → 避免 GBK 解码失败
    exit_code = os.system(cmd_str)

    # 检查是否成功
    if exit_code == 0:
        print("\n✅ 评估成功完成！")
        # 自动读取 results.csv 提取指标（无需捕获 stdout）
        results_csv = Path("runs/val/eval_vis_onshore_baseline/results.csv")
        if results_csv.exists():
            import pandas as pd
            try:
                df = pd.read_csv(results_csv)
                # Ultralytics v8.3+ 的 metrics 列名带 (B)
                mAP50 = df['metrics/mAP50(B)'].iloc[-1]
                mAP5095 = df['metrics/mAP50-95(B)'].iloc[-1]
                precision = df['metrics/precision(B)'].iloc[-1]
                recall = df['metrics/recall(B)'].iloc[-1]
                print(f"\n🎯 最终评估结果:")
                print(f"   mAP50       = {mAP50:.4f}")
                print(f"   mAP50-95    = {mAP5095:.4f}")
                print(f"   Precision   = {precision:.4f}")
                print(f"   Recall      = {recall:.4f}")
                print(f"\n📊 结果已保存至: {results_csv.parent.resolve()}")
            except Exception as e:
                print(f"⚠️  无法解析 results.csv（但评估已运行）: {e}")
        else:
            print(f"⚠️  results.csv 未生成，请检查 runs/val/ 路径")
    else:
        print(f"\n❌ yolo val 失败，退出码: {exit_code}")
        return False


if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 70)
    if success:
        print("🎉 全流程成功完成！")
        print("📌 提示：该结果即为「文献原模型」在海上视频上的 baseline 性能")
    else:
        print("❌ 流程中断，请检查日志")
    print("=" * 70)