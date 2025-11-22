import os

frames_dir = r"D:\VOD_Data\Videos_frames"
if not os.path.exists(frames_dir):
    print(f"❌ 文件夹 {frames_dir} 不存在！")
else:
    files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    print(f"✅ 找到 {len(files)} 张图片")

# 可选：打印前 5 个文件名
for i, f in enumerate(files[:5]):
    print(f"  {i+1}: {f}")