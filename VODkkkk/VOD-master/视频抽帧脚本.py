from PIL import Image
import cv2
import os

video_dir = r"D:\广财\F's research\ShipsVideos\NIR\Videos"
output_dir = r"D:\广财\F's research\ShipsVideos\NIR\Videos_frames"

os.makedirs(output_dir, exist_ok=True)

for video_name in os.listdir(video_dir):
    if not video_name.endswith((".mp4", ".avi")):
        continue

    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_name = f"{video_name.replace('.mp4', '').replace('.avi', '')}_frame{frame_id:06d}.jpg"
        img_path = os.path.join(output_dir, img_name)

        # ✅ 用 PIL 保存图片（支持中文路径）
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_img.save(img_path, quality=95)
        except Exception as e:
            print(f"❌ 保存失败: {e} | 路径: {img_path}")

        frame_id += 1

    cap.release()
    print(f"✅ 已抽取 {video_name} 共 {frame_id} 帧")

print("🎉 所有视频抽帧完成！")